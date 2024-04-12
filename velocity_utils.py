import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os
import time
import logging
import cv2
import matplotlib
import torchvision.transforms as f
from torchvision.transforms import InterpolationMode
from torch.cuda import amp

def mean(x):
    return sum(x)/len(x)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def reduce_tensor(tensor, world_size):
    '''
        for acc kind, get the mean in each gpu
    '''
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    # rt /= world_size
    return rt

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def project(Xs, Ks):
    '''
    Xs: [B, T, H, W, 3, 1]
    Ks: [B, T, 3, 3]
    '''
    Ks = Ks[:, :, None, None]
    grid = (Ks @ Xs).squeeze(-1)
    grid[..., :2] = grid[..., :2] / grid[..., -1:] # [B, T, H, W, 2]
    return grid[..., :2]
    # X, Y, Z = Xs.unbind(dim=-1)
    # fx, fy, cx, cy =\
    #     intrinsics[:, 0:1, 0], intrinsics[:, 1:2, 1], intrinsics[:, 0:1, 2], intrinsics[:, 1:2, 2]

    # x = fx * (X / Z) + cx
    # y = fy * (Y / Z) + cy

    # coords = torch.stack([x, y], dim=-1)
    # return coords

def inv_project(depths, intrinsics, h, w, grid, ones):
    '''
    depths: [B, T, H, W]
    intrinsics: [B, T, 3, 3]
    '''
    B, T, H, W = depths.shape

    grid[..., 0] = w.view(1, 1, 1, W)
    grid[..., 1] = h.view(1, 1, H, 1)
    grid = torch.cat((grid, ones), dim=-1).unsqueeze(-1)  # [B, T, H, W, 3, 1]
    inv_Ks = torch.inverse(intrinsics)[:, :, None, None]  # [B, T, 1, 1, 3, 3]

    grid = grid * depths[..., None, None]
    pts = inv_Ks @ grid # [B, T, H, W, 3, 1]
    return pts


def novel2ref(rts_novel, rts_ref):
    R_ref, T_ref = rts_ref[..., :3], rts_ref[..., -1:]
    invR_t = rts_novel[..., :3].permute(0, 2, 1)
    T_t = rts_novel[..., -1:]
    R = R_ref @ invR_t
    T = -R @ T_t + T_ref
    return torch.cat((R, T), dim=-1)

def dot(x, norm):
    # x: [N, 15, 3]
    # norm: [15, 3]
    length = torch.sum(x * norm, dim=-1)
    return torch.mean(torch.abs(length))

def select_in_list(select_list, mask):
    out = []
    for item in select_list:
        out.append(item[mask])
    return out

def squeeze_list(x):
    out = []
    for item in x:
        out.append(item[-1])
    return out

def embed(inputs, N_freqs):
    assert inputs.shape[-1] == 3
    freq_bands = 2.**torch.linspace(0., N_freqs-1, steps=N_freqs).to(inputs.device)
    repeat_ = inputs.dim()-1
    inputs_scaled = (inputs.unsqueeze(-2) * freq_bands.view(*[1]*repeat_,-1,1)).reshape(*inputs.shape[:-1],-1)
    inputs_scaled = torch.cat((inputs, torch.sin(inputs_scaled), torch.cos(inputs_scaled)),dim=-1)
    return inputs_scaled

### ref https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py ###
def knn_rays(nsample, rays, pts):
    """
    Input:
        nsample: max sample number in local region
        rays: 2 * [B, N, 3]
        pts: [B, H, W, 3]
    Return:
        group_idx: grouped points index, [B, N, nsample]
    """
    sqrdists = square_distance(rays, pts)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def square_distance(rays, pts):
    """
    Input:
        rays: 2 * [B, N, 3]
        pts: [B, H, W, 3]
    Output:
        dist: [B, N, H*W]
    """
    x1, x2 = rays
    x1 = x1.unsqueeze(-2) # [B, N, 1, 3]
    x2 = x2.unsqueeze(-2) # [B, N, 1, 3]
    pts = pts.flatten(1, 2).unsqueeze(1) # [B, 1, H*W, 3]
    cross = torch.norm(torch.cross(x2-x1, x1-pts, dim=-1), dim=-1)  # [B, N, H*W]
    dist = cross / torch.norm(x2-x1, dim=-1) # [B, N, H*W]
    return dist

def sample_rays(masks, n_sample, Ks, rts, val=False):
    '''
    masks: [B, H, W]
    Ks: [B, 1, 3, 3]
    rts: [B, 3, 4]
    '''
    B, H, W = masks.shape
    idx = []
    for i in range(B):
        coords = torch.stack((torch.where(masks[i].float() >= 0.5)), -1) # [N_dy, 2]
        batch_idx = i*torch.ones(coords.shape[0], 1).to(coords.device) # [N_dy, 1]
        coords = torch.cat((batch_idx, coords), dim=-1) # [N_dy, 3]
        if val:
            return coords
        select_inds = np.random.choice(coords.shape[0], size=[min(len(coords), n_sample)], replace=False)
        select_coords = coords[select_inds] # [n_sample, 3]
        idx.append(select_coords.unsqueeze(0))
    idx = torch.cat(idx, dim=0).reshape(-1, 3).long() # [B * n_sample, 3]

    inv_R, T = rts[..., :3].transpose(-1, -2), rts[..., -1:]
    rays_o = -inv_R @ T # [B, 3, 1]
    rays_o = rays_o.unsqueeze(1).repeat(1, n_sample, 1, 1)
    rays_d = inv_project(torch.ones(B, 1, H, W).to(coords.device), Ks).squeeze(1) # [B, H, W, 3, 1]
    rays_d = rays_d[idx[:, 0], idx[:, 1], idx[:, 2]].reshape(B, n_sample, 3, 1) # [B, n_sample, 3, 1]
    rays_d = inv_R[:, None, ...] @ rays_d
    rays_d = rays_d.squeeze(-1) / torch.norm(rays_d, dim=-2) # [B, n_sample, 3]
    rays_o = rays_o.squeeze(-1) # [B, n_sample, 3]
    return [rays_o, rays_d], idx

def get_2d_traj(ref, traj, rts, Ks):
    '''
    ref: N, 3
    traj: 6, N, 3
    rts_ref: 3, 4
    rts_target: 6, 3, 4
    K: 7, 3, 3
    '''
    ref_idx = traj.shape[0] // 2
    N = traj.shape[1]
    
    ref_cam = rts[ref_idx, :, :3] @ ref[..., None] + rts[ref_idx, :, -1:] # [N, 3, 1]
    ref_2d = project(ref_cam.reshape(1, 1, 1, N, 3, 1),\
                      Ks[None, ref_idx:ref_idx+1]).reshape(1, 1, N, 2) # [1, 1, N, 2]

    rts_target = torch.cat((rts[:ref_idx], rts[ref_idx+1:]), dim=0)
    traj_cam = rts_target[:, None, :, :3] @ traj[..., None] + rts_target[:, None, :, -1:] # [6, N, 3, 1]
    Ks_target = torch.cat((Ks[:ref_idx], Ks[ref_idx+1:]), dim=0)
    traj_2d = project(traj_cam.reshape(1, -1, 1, N, 3, 1),\
                       Ks_target[None, ...]).reshape(-1, 1, N, 2) # [6, 1, N, 2]

    traj_b = [ref_2d]
    for tt in range(ref_idx):
        traj_b.append(traj_2d[ref_idx-1-tt].unsqueeze(0)) # [1, 1, N, 2]

    traj_b = torch.stack(traj_b, dim=1) # [1, 4, N, 2]
    traj_f = traj_2d[-ref_idx:, 0].unsqueeze(0) # [1, 3, N, 2]
    traj_f = torch.cat((ref_2d, traj_f), dim=1) # [1, 4, N, 2]

    return traj_b.reshape(1, -1, N, 2), traj_f.reshape(1, -1, N, 2)

def resize_trajs(trajs, H_scale, W_scale):
    out = []
    for traj in trajs:
        traj[..., 0] *= W_scale
        traj[..., 1] *= H_scale
        out.append(traj)
    return out

def aggregate_features(flow, maps, h, w, grid, zeros):
    B, T, H, W, _ = maps.shape
    grid[..., 0] = w.view(1, 1, 1, W)
    grid[..., 1] = h.view(1, 1, H, 1)
    grid = grid + flow

    grid[..., 0] = grid[..., 0] / ((W-1)/2) - 1
    grid[..., 1] = grid[..., 1] / ((H-1)/2) - 1  # [B, T, H, W, 2]

    masks = (grid >= -1.0) * (grid <= 1.0)
    masks = masks[..., 0] * masks[..., 1] # [B, T, H, W]
    masks = torch.sum(masks, dim=1) # [B, H, W]
    masks = torch.where(masks == 0, zeros, 1./masks) # [B, H, W]

    maps = maps.flatten(0, 1).permute(0, 3, 1, 2)  # [B*T, C, H, W]
    # warp_maps = F.grid_sample(maps.float(), grid.flatten(0, 1).float(), mode='bilinear', padding_mode="zeros", align_corners=True) # [B*T, C, H, W]
    warp_maps = F.grid_sample(maps.float(), grid.flatten(0, 1).float(), mode='nearest', padding_mode="zeros", align_corners=True) # [B*T, C, H, W]
    warp_maps = warp_maps.reshape(B, T, -1, H, W)
    feat_map = warp_maps.sum(1).permute(0, 2, 3, 1) * masks[..., None]
    return feat_map

def resize_data(images, Ks, depths, flows, size):
    
    B, _, T, H, W = images.shape

    images = images.permute(0, 2, 1, 3, 4).flatten(0, 1) # [B*T, 3, H, W]
    depths = depths.reshape(B, T, 1, H, W).flatten(0, 1) # [B*T, 1, H, W]
    flows = flows.flatten(0, 1).permute(0, 3, 1, 2) # [B*T, 2, H, W]

    W_new, H_new = size
    W_scale = W_new / W
    H_scale = H_new / H
    resize_func = f.Resize(size=(H_new, W_new), interpolation=InterpolationMode.BILINEAR)

    images = resize_func(images)
    depths = resize_func(depths)
    flows = resize_func(flows)
    flows[:, 0] *= W_scale
    flows[:, 1] *= H_scale
    Ks[..., 0, :] *= W_scale
    Ks[..., 1, :] *= H_scale

    images = images.reshape(B, T, 3, H_new, W_new).permute(0, 2, 1, 3, 4) # [B, 3, T, H, W]
    depths = depths.reshape(B, T, H_new, W_new) # [B, T, H, W]
    flows = flows.reshape(B, T, 2, H_new, W_new).permute(0, 1, 3, 4, 2) # [B, T, H, W, 2]
    return images, Ks, depths, flows

def normalize(x):
  return x / np.linalg.norm(x)
def viewmatrix(z, up, pos):
  vec2 = normalize(z)
  vec1_avg = up
  vec0 = normalize(np.cross(vec1_avg, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, pos], 1)
  return m
def poses_avg(poses):
  center = poses[:, :3, 3].mean(0)
  vec2 = poses[:, :3, 2].sum(0)
  up = poses[:, :3, 1].sum(0)
  c2w = viewmatrix(vec2, up, center) # [3, 4]
  return c2w
def recenter_poses(poses):
  """Recenter camera poses into centroid."""
  poses_ = poses + 0
  bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
  c2w = poses_avg(poses)
  c2w = np.concatenate([c2w[:3, :4], bottom], -2) # [4, 4]
  bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
  poses = np.concatenate([poses[:, :3, :4], bottom], -2)

  poses = np.linalg.inv(c2w) @ poses
  poses_[:, :3, :4] = poses[:, :3, :4]
  poses = poses_
  return poses, c2w