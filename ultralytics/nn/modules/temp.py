from ultralytics.nn.modules import Conv, ContMixBlock, C2f_ContMix, C2f
import torch

x = torch.randn(1, 128, 16, 16)  # 假设输入尺寸
conv = Conv(128, 256, k=3, s=2, p=1)

print("Conv output shape:", conv(x).shape)

model = C2f_ContMix(c1=128, c2=128, n=2, shortcut=True)
model2 = C2f(c1=128, c2=128, n=2, shortcut=True)

y = model(x)
y2 = model2(x)
print(y.shape)  # 输出：[1, 128, 80, 80]
print(y2.shape)
