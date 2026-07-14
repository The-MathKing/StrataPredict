import torch
from dynamic_layers import LatentDynamicSimplicialConv

# Setup random dummy graph (10-line sanity check)
x = torch.randn(5, 16, requires_grad=True)
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
edge_attr = torch.randn(4, 1)

conv = LatentDynamicSimplicialConv(16, 32)
out = conv(x, edge_index, edge_attr)
print(f"Output shape: {out.shape}")

loss = out.sum()
loss.backward()
print(f"Gradients computed successfully! Beta grad: {conv.beta.grad}, w_curve grad shape: {conv.w_curve.grad.shape}")
