from ultralytics.nn.modules import Conv, SimAM, SPPF
import torch

x = torch.randn(1, 512, 20, 20)  # 假设输入尺寸
# conv = Conv(128, 256, k=3, s=2, p=1)
simam = SimAM(256)
sppf = SPPF(512, 512, k=5)
# print("Conv output shape:", conv(x).shape)
print("SimAM output shape:", simam(x).shape)
print("SPPF output shape:", sppf(x).shape)
