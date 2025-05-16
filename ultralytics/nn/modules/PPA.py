# --------------------------------------------------------
# 论文名称:HCF-Net: Hierarchical Context Fusion Network for Infrared Small Object Detection (arxiv 2024)
# 论文地址：https://arxiv.org/abs/2403.10778
# 代码地址:https://github.com/zhengshuchen/HCFNet
# ------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        # 定义一个卷积层，用于计算空间注意力
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # 定义一个Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算输入特征图的平均值
        avgout = torch.mean(x, dim=1, keepdim=True)
        # 计算输入特征图的最大值
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        # 将平均值和最大值在通道维度上拼接
        out = torch.cat([avgout, maxout], dim=1)
        # 通过卷积层和Sigmoid激活函数计算空间注意力权重
        out = self.sigmoid(self.conv2d(out))
        # 将注意力权重应用到输入特征图上
        return out * x


class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
        super().__init__()
        # 定义一个1x1卷积层，用于跳连接
        self.skip = conv_block(in_features=in_features,
                               out_features=filters,
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               norm_type='bn',
                               activation=False)
        # 定义三个3x3卷积层
        self.c1 = conv_block(in_features=in_features,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c2 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c3 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        # 定义空间注意力模块
        self.sa = SpatialAttentionModule()
        # 定义ECA模块
        self.cn = ECA(filters)
        # 定义两个不同patch size的局部全局注意力模块
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)

        # 定义批量归一化层
        self.bn1 = nn.BatchNorm2d(filters)
        # 定义dropout层
        self.drop = nn.Dropout2d(0.1)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()

        # 定义GELU激活函数
        self.gelu = nn.GELU()

    def forward(self, x):
        # 计算跳连接
        x_skip = self.skip(x)
        # 计算两个不同patch size的局部全局注意力
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        # 通过三个3x3卷积层
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        # 将三个卷积层的输出、跳连接和两个局部全局注意力的输出相加
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        # 通过ECA模块
        x = self.cn(x)
        # 通过空间注意力模块
        x = self.sa(x)
        # 通过dropout层
        x = self.drop(x)
        # 通过批量归一化层
        x = self.bn1(x)
        # 通过ReLU激活函数
        x = self.relu(x)
        return x


class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        # 定义两个全连接层
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        # 定义一个1x1卷积层
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        # 定义一个可学习的prompt向量
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        # 定义一个可学习的top-down变换矩阵
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        # 将特征图从(B, C, H, W)转换为(B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # 提取局部patch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P * P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        # 通过两个全连接层
        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        # 计算局部注意力
        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention  # (B, H/P*W/P, output_dim)

        # 计算cosine similarity并应用mask
        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # 恢复特征图形状
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output


class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        # 计算卷积核大小
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        # 定义自适应平均池化层
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 定义一个1D卷积层
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算全局平均池化
        out = self.pool(x)
        # 调整特征图形状
        out = out.view(x.size(0), 1, x.size(1))
        # 通过1D卷积层和Sigmoid激活函数
        out = self.conv(out)
        # 调整特征图形状
        out = out.view(x.size(0), x.size(1), 1, 1)
        # 将全局注意力权重应用到输入特征图上
        return out * x


class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1
                 ):
        super().__init__()
        # 定义卷积层
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


if __name__ == '__main__':
    block = PPA(in_features=64, filters=64)  # 输入通道数，输出通道数
    input = torch.rand(3, 64, 128, 128)  # 输入 B C H W
    output = block(input)
    print(input.size())
    print(output.size())

