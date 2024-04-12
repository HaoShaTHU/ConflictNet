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

# value_test = torch.tensor([1, 2, 2, 7, 9, 3, 6, 5, 8, 4, 1, 7]).reshape(1, -1)
# adj_test = torch.tensor([
#   [0, 0, 0, 1],
#   [1, 4, 0, 2],
#   [2, 5, 1, 3],
#   [3, 6, 2, 3],
#   [1, 7, 4, 5],
#   [2, 8, 4, 6],
#   [3, 9, 5, 6],
#   [4, 7, 7, 8],
#   [5, 10, 7, 9],
#   [6, 9, 8, 9],
#   [8, 11, 10, 10],
#   [10, 11, 11, 11]
#   ]).reshape(1, -1, 4)

# out = laplace(value_test, adj_test, 1, 1)
# print(out)