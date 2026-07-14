import torch
from dynamic_layers import LatentDynamicSimplicialConv

def test_dynamic_conv():
    # Setup reproducible dummy data
    torch.manual_seed(42)
    num_nodes = 4
    in_channels = 16
    out_channels = 32
    
    # 4 nodes, 4 edges
    x = torch.randn(num_nodes, in_channels)
    
    # Simple line graph: 0-1-2-3 + triangle 1-2-3 (let's just make arbitrary edges)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long)
    
    num_edges = edge_index.size(1)
    
    # Static topology: 4 - deg(u) - deg(v) + 3 * num_triangles(e)
    # Randomly assign for test
    edge_attr = torch.randn(num_edges, 1)
    
    # Instantiate Layer
    conv = LatentDynamicSimplicialConv(in_channels, out_channels)
    
    # Check learnable parameters
    assert conv.beta.requires_grad == True
    assert conv.w_curve.requires_grad == True
    
    # Forward Pass
    out = conv(x, edge_index, edge_attr)
    
    # Assert dimensions
    assert out.shape == (num_nodes, out_channels), f"Expected shape {(num_nodes, out_channels)}, got {out.shape}"
    
    # Test gradient flow (Backprop)
    loss = out.sum()
    loss.backward()
    
    # Verify gradients
    assert conv.beta.grad is not None, "Gradient did not flow to beta!"
    assert conv.w_curve.grad is not None, "Gradient did not flow to w_curve!"
    assert conv.lin_feat.weight.grad is not None, "Gradient did not flow to linear layer!"
    
    print("ALL TESTS PASSED: Forward Pass successful. Tensor dimensions correct. Gradients flowing back to beta and w_curve.")

if __name__ == "__main__":
    test_dynamic_conv()
