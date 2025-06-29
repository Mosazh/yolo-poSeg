import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv

# 简化版DynamicConv示例（多个卷积核按权重加权）
class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, K=4, reduction=4):
        super().__init__()
        self.K = K  # 动态卷积核数量
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 固定多个卷积核组
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
            for _ in range(K)
        ])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, K)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        # 生成权重
        w_ = self.gap(x).view(b, c)
        w_ = self.fc(w_)
        w_ = self.softmax(w_)  # b x K 权重分布

        # 分别计算所有卷积结果
        conv_outs = [conv(x) for conv in self.convs]  # K x (b,c,h,w)
        conv_outs = torch.stack(conv_outs, dim=1)  # b x K x c x h x w

        # 按权重加权求和
        w_ = w_.view(b, self.K, 1, 1, 1)
        out = (conv_outs * w_).sum(dim=1)  # b x c x h x w
        return out


# ECA 注意力模块，参数量极小
class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # b x c x 1 x 1
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)

class DynamGSConv(nn.Module):
    def __init__(self, c1, c2, kernel_size=1, stride=1, groups=1, act=True, dynamic_K=4):
        super().__init__()
        c_ = c2 // 2
        # 1x1 分组卷积压缩通道
        self.cv1 = Conv(c1, c_, kernel_size, stride, None, groups, 1, act)
        # 用动态深度卷积替代固定depthwise conv
        self.dynamic_conv = DynamicConv(c_, c_, kernel_size=5, stride=1, padding=2, K=dynamic_K)
        # 1x1 卷积融合两部分特征，相当于可学习的 shuffle
        self.channel_mix = nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
        self.eca = ECA(c2)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.dynamic_conv(x1)
        out = torch.cat([x1, x2], dim=1)
        out = self.channel_mix(out)
        out = self.bn(out)
        out = self.eca(out)
        out = self.act(out)
        return out

if __name__ == "__main__":
    model = DynamGSConv(64, 128, 3, 2, 1)
    x = torch.randn(1, 64, 56, 56)
    out = model(x)
    conv = Conv(64, 128, 3, 2, 1)
    print(out.shape)  # torch.Size([1, 128, 56, 56])
    print(conv(x).shape)
