import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import clever_format, profile
# from torchsummary import summary


# https://link.springer.com/chapter/10.1007/978-3-031-72114-4_60
# TinyU-Net: Lighter Yet Better U-Net with Cascaded Multi-receptive Fields

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """自动计算并返回填充值，以确保卷积后的输出大小与输入一致 ('same' padding)。"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # 考虑膨胀（dilation）时计算实际的卷积核大小
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 如果未提供填充值，默认根据卷积核大小自动计算填充值
    return p


class Conv(nn.Module):
    """标准卷积模块，包含卷积层、批归一化层和激活函数。"""

    default_act = nn.SiLU()  # 默认激活函数为SiLU（Sigmoid Linear Unit）

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """初始化卷积层，包含卷积核大小、步幅、填充、分组卷积、膨胀等参数。
        可选激活函数，可传入自定义激活函数或使用默认的SiLU激活函数。
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)  # 卷积层
        self.bn = nn.BatchNorm2d(c2)  # 批归一化层，帮助加速训练并稳定网络
        # 如果传入的激活函数为True，则使用默认的SiLU激活函数，否则使用自定义激活函数或无激活函数
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """前向传播：首先经过卷积层、批归一化层，然后应用激活函数。"""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """前向传播（融合版本）：跳过批归一化，只进行卷积和激活，适用于某些优化情境。"""
        return self.act(self.conv(x))


class DWConv(Conv):
    """深度可分离卷积（Depth-wise Convolution），每个输入通道使用独立卷积核进行卷积。"""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """初始化深度可分离卷积层，与标准卷积的区别在于使用了深度可分离卷积。
        深度可分离卷积减少了参数量，适用于轻量化网络。
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)  # 通过gcd确定groups参数


# 轻量级级联多接收域模块（Lightweight Cascade Multi-Receptive Fields Module）
class CMRF(nn.Module):
    """CMRF模块，支持多个输入输出通道数，使用级联结构结合多个接收域（receptive fields）。"""

    def __init__(self, c1, c2, N=8, shortcut=True, g=1, e=0.5):
        """初始化CMRF模块。
        N：模块中的深度卷积数量，默认8。
        shortcut：是否使用shortcut连接，即直接将输入加到输出中（默认True）。
        g：分组卷积的分组数，通常为1。
        e：通道扩展因子，默认0.5。
        """
        super().__init__()

        self.N = N  # 模块的深度卷积数量
        self.c = int(c2 * e / self.N)  # 计算每个深度卷积的输出通道数
        self.add = shortcut and c1 == c2  # 如果需要shortcut连接且输入输出通道相同，则启用shortcut连接

        # pwconv1：1x1卷积，用于通道压缩
        self.pwconv1 = Conv(c1, c2 // self.N, 1, 1)
        # pwconv2：1x1卷积，用于恢复通道数
        self.pwconv2 = Conv(c2 // 2, c2, 1, 1)
        # 深度卷积（Depthwise Convolution），生成多个深度卷积层
        self.m = nn.ModuleList(DWConv(self.c, self.c, k=3, act=False) for _ in range(N - 1))

    def forward(self, x):
        """前向传播：首先经过1x1卷积压缩通道，然后经过N个深度卷积，最后将结果进行通道恢复。"""
        x_residual = x  # 保存输入，稍后可能会加回
        x = self.pwconv1(x)  # 经过pwconv1压缩通道

        # 将输入x拆分为两部分，分别进行处理并扩展
        x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]  # 拆分为奇数和偶数通道
        # 将x中的每一部分通过N个深度卷积进行处理
        x.extend(m(x[-1]) for m in self.m)
        x[0] = x[0] + x[1]  # 将两个部分相加
        x.pop(1)  # 删除第二部分

        # 将所有的输出连接在一起
        y = torch.cat(x, dim=1)
        y = self.pwconv2(y)  # 通过pwconv2恢复通道数
        return x_residual + y if self.add else y  # 如果使用shortcut连接，则加上输入，否则直接返回输出


if __name__ == '__main__':


    # 定义输入张量的形状为 [16, 512, 32, 32]
    input_tensor = torch.randn(1, 128, 32, 32)

    # 创建 ConvolutionalGLU 模块
    conv_glu = CMRF(128, 256)

    # 输入特征图的高度和宽度
    H = 32
    W = 32

    # 前向传播
    output = conv_glu(input_tensor)
    print("Output shape:", output.shape)
