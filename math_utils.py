import torch
import torch.nn as nn

def get_xy(input, adj):
  '''
  input: [B, N, C]
  adj: [N, 4]
  return x, y: [B, N, 2, C]
  '''
  value = input.clone()
  x = value[:, adj[:, 2:]].clone()
  y = value[:, adj[:, :2]].clone()
  return x, y

def laplace(input, adj, dx, dy):
  '''
  input: [B, N, C]
  adj: [N, 4]
  return out: [B, N, C]
  '''
  x, y = get_xy(input, adj)
  # breakpoint()
  x = (x.sum(dim=-2) - 2 * input) / (dx ** 2)
  y = (y.sum(dim=-2) - 2 * input) / (dy ** 2)
  out = x + y
  return out

def grad_dot(x_in, y_in, adj, dx, dy):
  '''
  x_in, y_in: [B, N, C]
  adj: [N, 4]
  return out: [B, N, C]
  '''
  x, _ = get_xy(x_in, adj)
  _, y = get_xy(y_in, adj)
  x = (x[..., 1, :] - x[..., 0, :]) / (2 * dx)
  y = (y[..., 1, :] - y[..., 0, :]) / (2 * dy)
  out = x + y
  return out

def grad(input, adj, dx, dy):
  '''
  input: [B, N, C]
  adj: [B, N, 4]
  return x, y: [B, N, C]
  '''
  x, y = get_xy(input, adj)
  x = (x[..., 1, :] - x[..., 0, :]) / (2 * dx)
  y = (y[..., 1, :] - y[..., 0, :]) / (2 * dy)
  return x, y
