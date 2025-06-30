import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, ECA, CBAM, SEBlock


############################################
# 改进版 DynamicConv：多尺度 + depthwise/标准可切换
############################################
class DynamicConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=[3,5,7,3],
        stride=1,
        padding=None,
        reduction=4,
        use_depthwise=True
    ):
        super().__init__()
        self.K = len(kernel_sizes)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_depthwise = use_depthwise

        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            pad = k // 2 if padding is None else padding
            groups = in_channels if use_depthwise else 1
            self.convs.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    stride=stride,
                    padding=pad,
                    groups=groups,
                    bias=False
                )
            )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, self.K)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        w_ = self.gap(x).view(b, c)
        w_ = self.fc(w_)
        w_ = self.softmax(w_)

        conv_outs = [conv(x) for conv in self.convs]
        conv_outs = torch.stack(conv_outs, dim=1)
        w_ = w_.view(b, self.K, 1, 1, 1)
        out = (conv_outs * w_).sum(dim=1)
        return out

############################################
# 改进版 DynamGSConv：支持指定注意力类型
############################################
class DynamGSConv(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        kernel_size=1,
        stride=1,
        groups=1,
        act=True,
        attype='eca',          # 可选：'eca' / 'se' / 'cbam'
        dynamic_kernel_sizes=[3,5,7,3],
        dynamic_reduction=4,
        use_dynamic_depthwise=True,
        eca_k_size=5,
        cbam_k_size=7,
        se_reduction=16
    ):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, kernel_size, stride, None, groups, 1, act)

        self.dynamic_conv = DynamicConv(
            in_channels=c_,
            out_channels=c_,
            kernel_sizes=dynamic_kernel_sizes,
            stride=1,
            reduction=dynamic_reduction,
            use_depthwise=use_dynamic_depthwise
        )

        self.channel_mix = nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

        # 按 attype 指定注意力模块
        if attype == 'eca':
            self.att = ECA(c2, k_size=eca_k_size)
        elif attype == 'se':
            self.att = SEBlock(c2, reduction=se_reduction)
        elif attype == 'cbam':
            self.att = CBAM(c2, kernel_size=cbam_k_size)
        else:
            self.att = nn.Identity()

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.dynamic_conv(x1)
        out = torch.cat([x1, x2], dim=1)
        out = self.channel_mix(out)
        out = self.bn(out)
        out = self.att(out)
        out = self.act(out)
        return out


############################################
# 测试
############################################
if __name__ == "__main__":
    x = torch.randn(1, 64, 56, 56)

    print("\n=== ECA版本 ===")
    model_eca = DynamGSConv(
        64, 128,
        kernel_size=3,
        stride=2,
        attype='eca'
    )
    print(model_eca(x).shape)

    print("\n=== SE版本 ===")
    model_se = DynamGSConv(
        64, 128,
        kernel_size=3,
        stride=2,
        attype='se',
        se_reduction=8
    )
    print(model_se(x).shape)

    print("\n=== CBAM版本 ===")
    model_cbam = DynamGSConv(
        64, 128,
        kernel_size=3,
        stride=2,
        attype='cbam',
        cbam_k_size=7
    )
    print(model_cbam(x).shape)
