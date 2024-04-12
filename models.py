import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.layers import drop_path, to_2tuple
import torch.nn.functional as F
import time
import math
from math_utils import laplace, grad_dot, grad
from base_models import MLP_decoder, GCN

class ConflictNet(nn.Module):
    def __init__(self,
                 device,
                 indims=35,
                 embeddim=64,
                 dydim=12,
                 stdim=6,
                 diffuse_step=30,
                 edge_num=25,
                 pred_window=1
                 ):
        super().__init__()
        
        self.indims = indims
        self.embeddim = embeddim
        self.edge_num = edge_num
        self.dydim = dydim
        self.stdim = stdim
        self.diffuse_step = diffuse_step
        self.device = device
        self.pred_window = pred_window

        self.death_encoder_short = GCN_LSTM(dims=[1*2, embeddim, embeddim//2])
        self.st_feat_encoder_short = GCN(dims=[indims*2, embeddim, embeddim//2])
        self.death_encoder_long = GCN_LSTM(dims=[1*2, embeddim, embeddim//2])
        self.st_feat_encoder_long = GCN(dims=[indims*2, embeddim, embeddim//2])
        self.cis = CIS_module(embeddim*2, embeddim*2, embeddim*2)
        self.decoder1 = MLP_decoder(dims=[embeddim*2, embeddim, embeddim//2, embeddim//4, dydim+stdim+1])

        self.trans_net = MLP_decoder(dims=[dydim*(2+1+1)+stdim*(2)+(2), embeddim, embeddim//2, 25+1])
        self.q_net = MLP_decoder(dims=[dydim*(2)+stdim*(2)+(2), embeddim, embeddim//2, dydim])

        self.decoder2 = MLP_decoder(dims=[dydim+stdim, embeddim//2, embeddim//2, embeddim, 1])

        self.D = nn.Parameter(torch.ones(dydim//3*2)*5e-2, requires_grad=True)
        self.mu = nn.Parameter(torch.ones(dydim//3*2)*5e-2, requires_grad=True)
        self.coeff = nn.Parameter(torch.ones(dydim**2-2*(dydim//3)**2-dydim)*5e-2, requires_grad=True)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.fill_(0)
            m.bias.data.fill_(0)

    def process_param(self, B, N):
        # coeff
        matrixs33 = []
        dim_single = self.dydim // 3
        dim33 = dim_single ** 2
        for i in range(4):
            matrixs33.append(self.coeff[i*dim33:(i+1)*dim33].reshape(dim_single, dim_single))
        last_idx = 4 * dim33
        matrixs_diag = []
        dim_diag = dim_single ** 2 - dim_single
        for i in range(3):
            aij = self.coeff[last_idx+i*dim_diag:last_idx+(i+1)*dim_diag].reshape(dim_single-1, dim_single)
            m = torch.cat(
                (torch.ones((dim_single-1, 1), device=self.device),
                 aij), dim=1) # [n-1, n+1]
            m = torch.cat((m.flatten(), torch.ones((1), device=self.device)))
            m = m.reshape(dim_single, dim_single)
            matrixs_diag.append(m)

        coeff1 = torch.cat((matrixs_diag[0], torch.zeros((dim_single, dim_single), device=self.device), matrixs33[0]), dim=1)
        coeff2 = torch.cat((torch.zeros((dim_single, dim_single), device=self.device), matrixs_diag[1], matrixs33[1]), dim=1)
        coeff3 = torch.cat((matrixs33[2], matrixs33[3], matrixs_diag[2]), dim=1)
        coeff = torch.cat((coeff1, coeff2, coeff3), dim=0)
        coeff = F.relu(coeff).reshape(1, 1, self.dydim, self.dydim)

        # param
        Dij = torch.cat((torch.zeros((dim_single), device=self.device), self.D))
        MUij = torch.cat((
            self.mu[:dim_single],
            torch.zeros((dim_single), device=self.device),
            self.mu[-dim_single:]))
        Dij = F.relu(Dij).reshape(1, 1, -1).repeat(B, N, 1)
        MUij = F.relu(MUij).reshape(1, 1, -1).repeat(B, N, 1)

        return coeff, Dij, MUij
    
    def process_adj(self, adj):
        N = adj.shape[0]
        arange = torch.arange(N).reshape(N, 1).repeat(1, 4).to(adj.device)
        adj4_uniq = torch.stack(
            (torch.max(arange, adj), torch.min(arange, adj)),
            dim=-1).reshape(-1, 2) # [N*4, 2]
        adj_uniq, inv_idx = torch.unique(adj4_uniq, dim=0, return_inverse=True) # [m, 2]
        return adj_uniq, inv_idx
    
    def cal_overlap_length(self, trans_dis, grid):
        min_trans = trans_dis - 1 / 2
        max_trans = trans_dis + 1 / 2
        min_grid = grid - 1 / 2
        max_grid = grid + 1 / 2
        overlap_min = F.relu(min_trans-min_grid) + min_grid # max
        overlap_max = -F.relu(max_trans-max_grid) + max_trans # min
        length = F.relu(overlap_max - overlap_min) # [B, N, 25, C]
        return length
    
    def get_trans_probs(self, delta_x, delta_y):     
        x = torch.arange(5)
        y = torch.arange(5)
        x, y = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([x, y], dim=-1).flatten(0, 1) # [25, 2]
        grid = grid.reshape(1, 1, 25, 2).to(delta_x.device)

        x = self.cal_overlap_length(delta_x[..., None, :], grid[..., 1:2])
        y = self.cal_overlap_length(delta_y[..., None, :], grid[..., 0:1])
        area = x * y # [B, N, 25, C]

        area = torch.cat(
            (area[..., :12, :], area[..., 12:13, :], area[..., 13:, :]), dim=-2)
        return area
    
    def pde(self, phi, adj4, adj25, feat_dy, feat_st):
        '''
        phi: [B, N]
        adj4: [B, N, 4]
        death: [B, N, 2]
        '''
        dx = dy = 0.5
        dt = 1
        B, N, C = feat_dy.shape[:3]
        feat = feat_dy

        coeff, Dij, MUij = self.process_param(B, N)

        us = []
        mix_probs_month = []
        q0_month = []
        feat_df = []
        for i in range(1, self.diffuse_step*self.pred_window+1):
            last_feat = feat.clone() # [B, N, C]
            
            # diffusion
            J = laplace(feat, adj4[0], dx, dy) # [B, N, C]
            grad_phi_x, grad_phi_y = grad(phi[..., None], adj4[0], dx, dy)
            F = grad_dot(grad_phi_x*feat, grad_phi_y*feat, adj4[0], dx, dy) # [B, N, C]
            u_diffuse = Dij * J - MUij * F # [B, N, C]
            u_diffuse = (u_diffuse[..., None, :] @ coeff).squeeze(-2) # [B, N, C]

            # rigid motion
            mass = torch.where(
                last_feat == 0,
                torch.ones_like(last_feat).to(last_feat.device).float(),
                last_feat
                )
            a_x = ((MUij * grad_phi_x)[..., None, :] @ coeff).squeeze(-2) / mass # [B, N, C]
            a_y = ((MUij * grad_phi_y)[..., None, :] @ coeff).squeeze(-2) / mass # [B, N, C]
            # we only allow the dy_feats to move in a 5*5 area
            a_x = torch.clip(a_x, -2, 2)
            a_y = torch.clip(a_y, -2, 2)
            motion_probs = self.get_trans_probs(a_x*dt, a_y*dt) # [B, N, 25, C]
            u_rigid = torch.zeros((B, N, 25, C), device=feat.device)
            u_rigid.scatter_add_(dim=1, index=adj25[..., None].repeat(1, 1, 1, C), src=motion_probs*last_feat[..., None, :])
            u_rigid = u_rigid.sum(dim=-2) - last_feat # [B, N, C]

            # mix the diffusion and rigid motion
            feat_trans = torch.cat((
                self.aggre_st_feat(feat, adj25), # [B, N, C*2]
                self.aggre_st_feat(phi[..., None], adj25), # [B, N, 2]
                Dij.clone().detach(), # [B, N, C]
                MUij.clone().detach(), # [B, N, C]
                # v_diffuse.flatten(-2, -1), # [B, N, C*5]
                self.aggre_st_feat(feat_st, adj25), # [B, N, C*2]
                ), dim=-1)
            trans_out = self.trans_net(feat_trans)
            mix_probs = torch.sigmoid(trans_out[..., -1:]) # [B, N, 1]
            mix_probs_month.append(mix_probs[:, :, 0].clone())
            delta_u = mix_probs * u_diffuse + (1 - mix_probs) * u_rigid # [B, N, C]

            # predict q0
            feat_q = torch.cat((
                self.aggre_st_feat(last_feat, adj25), # [B, N, C*2]
                self.aggre_st_feat(phi[..., None], adj25), # [B, N, 2]
                self.aggre_st_feat(feat_st, adj25)), # [B, N, C*2]
                dim=-1)
            q0 = self.q_net(feat_q) # [B, N, C]
            q0_month.append(q0.clone())

            # unpdate feat
            feat = last_feat + delta_u + q0

            final_u = self.decoder2(torch.cat((feat, feat_st), dim=-1))
            feat_df.append(feat.clone())
            us.append(final_u.clone())

        us = torch.cat(us, dim=-1).reshape(B, N, self.pred_window, self.diffuse_step)
        mix_probs_month = torch.stack(mix_probs_month, dim=-1).reshape(B, N, self.pred_window, self.diffuse_step)
        q0_month = torch.stack(q0_month, dim=-1).reshape(B, N, -1, self.pred_window, self.diffuse_step)
        feat_df = torch.stack(feat_df, dim=-2).reshape(B, N, self.pred_window, self.diffuse_step, -1)
        return us, mix_probs_month, q0_month, feat_df
    
    def aggre_st_feat(self, input, adj):
        x = input[:, adj[0]].clone() # [B, N, 25, C]
        k = x.shape[-2]
        de = x.device
        weight = torch.cat(
            (torch.ones((1), device=de), torch.ones((k-1), device=de)*0.01))
        weight = weight.reshape(1, 1, -1, 1)
        x = (x * weight).sum(dim=-2)
        x = torch.cat((input, x), dim=-1)
        return x
    
    def aggre_dy_feat(self, input, adj):
        x = input[:, :, adj[0]].clone() # [B, T, N, 25, C]
        k = x.shape[-2]
        de = x.device
        weight = torch.cat(
            (torch.ones((1), device=de), torch.ones((k-1), device=de)*0.01))
        weight = weight.reshape(1, 1, 1, -1, 1)
        x = (x * weight).sum(dim=-2)
        x = torch.cat((input, x), dim=-1)
        return x
    
    def encode(self, feat, adj25, death_encoder, st_feat_encoder):
        feat_death = self.aggre_dy_feat(feat[..., :1], adj25)
        feat_st = self.aggre_st_feat(feat[..., 1:].mean(dim=1), adj25)
        feat_death = death_encoder(feat_death, adj25)
        feat_st = st_feat_encoder(feat_st, adj25)
        feat = torch.cat((feat_death, feat_st), dim=-1)
        return feat
    
    def decode(self, input_short, input_long, adj25, spatial_idx=None):
        feat_short = self.encode(input_short, adj25, self.death_encoder_short, self.st_feat_encoder_short)
        feat_long = self.encode(input_long, adj25, self.death_encoder_long, self.st_feat_encoder_long)
        feat = torch.cat((feat_short, feat_long), dim=-1)

        # debias
        if spatial_idx is not None:
            unbias_feat = []
            for b in range(feat.shape[0]):
                idx = spatial_idx[b]
                sums = torch.zeros(idx.max().item()+1, feat.shape[-1], device=feat.device)
                sums.scatter_add_(0, idx.unsqueeze(-1).expand(-1, feat.shape[-1]), feat[b])
                counts = idx.bincount(minlength=idx.max().item()+1).type_as(sums)
                spacial_feat = sums / (counts.unsqueeze(-1)+1) # [k, C]
                p = counts.unsqueeze(-1) / feat.shape[1] # [k, 1]
                unbias = self.cis(feat[b], [spacial_feat, p*spacial_feat])
                unbias_feat.append(unbias)
            feat = torch.stack(unbias_feat, dim=0)

        x = self.decoder1(feat)
        phi = x[..., -1].clone()
        feat_dy = x[..., :self.dydim].clone()
        feat_st = x[..., self.dydim:self.dydim+self.stdim].clone()
        return phi, feat_dy, feat_st
    
    def forward(self, input_short, adj, input_long, adj25, spatial_idx=None, sharpness=None):
        if sharpness is not None:
            self.sharpness = sharpness
        phi, feat_dy, feat_st = self.decode(input_short[:, :-1], input_long[:, :-self.diffuse_step], adj25, spatial_idx[:, 0])
        phi_next, _, _ = self.decode(input_short[:, 1:], input_long[:, self.diffuse_step:], adj25, spatial_idx[:, 1])

        us, mix_probs, q0, feat_df = self.pde(phi, adj, adj25, feat_dy, feat_st)

        probs_day = torch.sigmoid(us / self.sharpness) # [B, N, months, steps]
        probs_month = 1 - torch.prod(1-probs_day, dim=-1) # [B, N, months]
        return probs_month, phi, phi_next, probs_day, mix_probs, q0, feat_df


class GCN_LSTM(nn.Module):
    def __init__(self, dims, atten_depth=2):
        super(GCN_LSTM, self).__init__()

        self.gcn = GCN(dims)
        self.lstm = nn.LSTM(dims[-1], dims[-1], atten_depth, batch_first=True)

    def forward(self, x, adj):
        # [B, T, N, C]
        B, T, N, _ = x.shape
        x = self.gcn(x, adj)
        x = F.relu(x)
        x = x.permute(0, 2, 1, 3).flatten(0, 1) # [B*N, T, C]
        x, _ = self.lstm(x)
        x = x[:, -1].reshape(B, N, -1)
        return x

# debiasing module
# borrow from https://github.com/echoanran/CIS
class CIS_module(nn.Module):
    def __init__(self, input_channel, d_model, output_channel):
        super(CIS_module, self).__init__()
        
        self.input_channel = input_channel
        self.d_model = d_model
        self.output_channel = output_channel

        # subject attention
        self.kmap = nn.Linear(self.input_channel, self.d_model)
        self.qmap = nn.Linear(self.input_channel, self.d_model)
        self.xmap = nn.Linear(self.input_channel, self.output_channel)
        self.smap = nn.Linear(self.input_channel, self.output_channel)
   
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, x, subject_infos):
        # K_subject: s
        # V_subject: s*p

        K_subject, V_subject = subject_infos
        
        K_subject = self.kmap(K_subject)
        Q_subject = self.qmap(x)

        subject_embedding, _ = self.attention(Q_subject, K_subject, V_subject)
        x = self.xmap(x) + self.smap(subject_embedding)

        return x

