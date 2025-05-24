# https://github.com/icandle/MAN/blob/main/archs/MAN_arch.py

'''
This module enhances feature representation through a multi-scale attention mechanism while preserving the original information via residual connections.
The input and output shapes are consistent, making it suitable for stacking in deep networks.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class MLKA_Ablation(nn.Module):
    def __init__(self, n_feats, k=2, squeeze_factor=15):
        super().__init__()
        i_feats = 2*n_feats

        self.n_feats= n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        k = 2

        #Multiscale Large Kernel Attention
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats//k, n_feats//k, 7, 1, 7//2, groups= n_feats//k),
            nn.Conv2d(n_feats//k, n_feats//k, 9, stride=1, padding=(9//2)*4, groups=n_feats//k, dilation=4),
            nn.Conv2d(n_feats//k, n_feats//k, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats//k, n_feats//k, 5, 1, 5//2, groups= n_feats//k),
            nn.Conv2d(n_feats//k, n_feats//k, 7, stride=1, padding=(7//2)*3, groups=n_feats//k, dilation=3),
            nn.Conv2d(n_feats//k, n_feats//k, 1, 1, 0))
        '''self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats//k, n_feats//k, 3, 1, 1, groups= n_feats//k),
            nn.Conv2d(n_feats//k, n_feats//k, 5, stride=1, padding=(5//2)*2, groups=n_feats//k, dilation=2),
            nn.Conv2d(n_feats//k, n_feats//k, 1, 1, 0))'''

        #self.X3 = nn.Conv2d(n_feats//k, n_feats//k, 3, 1, 1, groups= n_feats//k)
        self.X5 = nn.Conv2d(n_feats//k, n_feats//k, 5, 1, 5//2, groups= n_feats//k)
        self.X7 = nn.Conv2d(n_feats//k, n_feats//k, 7, 1, 7//2, groups= n_feats//k)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))


    def forward(self, x, pre_attn=None, RAA=None):
        shortcut = x.clone()

        x = self.norm(x)

        x = self.proj_first(x)

        a, x = torch.chunk(x, 2, dim=1)

        #u_1, u_2, u_3= torch.chunk(u, 3, dim=1)
        a_1, a_2 = torch.chunk(a, 2, dim=1)

        a = torch.cat([self.LKA7(a_1)*self.X7(a_1), self.LKA5(a_2)*self.X5(a_2)], dim=1)

        x = self.proj_last(x*a)*self.scale + shortcut

        return x

if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.rand(1, 128, 160, 160)
    model = MLKA_Ablation(128)
    y = model(x)
    print(y.shape)
