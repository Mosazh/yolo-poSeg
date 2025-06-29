from ultralytics.nn.modules import RepGSConv
import torch

# 训练
m = RepGSConv(64, 128, s=1)
out = m(torch.randn(1, 64, 128, 128))
print(out.shape)

# 推理
# m.switch_to_deploy()
# out2 = m(torch.randn(1, 64, 128, 128))
# print(out2.shape)
