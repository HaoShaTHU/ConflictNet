import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.layers import drop_path, to_2tuple
from torch_geometric.nn import GATConv

class FC(nn.Module):
    def __init__(self, indim, outdim, dropout=0.):
        super(FC, self).__init__()
        self.fc = nn.Linear(indim, outdim)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, residual=None):
        x = self.fc(x)
        if residual is not None:
            x = x + residual
        x = F.gelu(x)
        x = self.dropout(x)
        return x

class MLP_decoder(nn.Module):
    def __init__(self, dims):
        super(MLP_decoder, self).__init__()

        self.fcs = nn.ModuleList(
                [FC(dims[i], dims[i+1]) for i in range(len(dims)-2)]
            )
        self.out_fc = nn.Linear(dims[-2], dims[-1])
    def forward(self, x):
        for i in range(len(self.fcs)):
            x = self.fcs[i](x)
        x = self.out_fc(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)
    
class GCN(nn.Module):
    def __init__(self, dims):
        super(GCN, self).__init__()

        self.fcs = nn.ModuleList(
                [FC(dims[i], dims[i+1]) for i in range(len(dims)-2)]
            )
        self.out_fc = nn.Linear(dims[-2], dims[-1])
    def forward(self, x, adj):
        for i in range(len(self.fcs)):
            x = self.fcs[i](x)
        x = self.out_fc(x)
        return x

def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return  torch.tensor(sinusoid_table,dtype=torch.float, requires_grad=False).unsqueeze(0) 

def sincos_embed(inputs, N_freqs):
    # inputs: [..., 3]
    # N_freqs: int
    assert inputs.shape[-1] == 2
    freq_bands = 2.**torch.linspace(0., N_freqs-1, steps=N_freqs).to(inputs.device)
    repeat_ = inputs.dim()-1
    inputs_scaled = (inputs.unsqueeze(-2) * freq_bands.view(*[1]*repeat_,-1,1)).reshape(*inputs.shape[:-1],-1)
    inputs_scaled = torch.cat((inputs, torch.sin(inputs_scaled), torch.cos(inputs_scaled)),dim=-1)
    return inputs_scaled


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)