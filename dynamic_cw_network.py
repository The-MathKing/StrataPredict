import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Data, Batch

from dynamic_layers import LatentDynamicSimplicialConv

class DynamicCWNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_classes):
        super(DynamicCWNetwork, self).__init__()
        
        # 1. Initial Continuous Embedding
        self.initial_embedding = nn.Linear(in_channels, hidden_channels)
        
        # 2. Topological Block (exactly 3 layers)
        self.conv1 = LatentDynamicSimplicialConv(hidden_channels, hidden_channels)
        self.conv2 = LatentDynamicSimplicialConv(hidden_channels, hidden_channels)
        self.conv3 = LatentDynamicSimplicialConv(hidden_channels, hidden_channels)
        
        # 4. Classification Head
        self.mlp_lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.dropout = nn.Dropout(p=0.5)
        self.mlp_lin2 = nn.Linear(hidden_channels // 2, out_classes)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # Project discrete/one-hot atom features into a continuous latent space
        x = F.elu(self.initial_embedding(x))
        
        # Apply the Topological Block
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        
        # 3. Global Readout Pooling (collapse node-level to graph-level)
        x = global_add_pool(x, batch)
        
        # Map pooled graph embedding to final output using the MLP
        x = F.elu(self.mlp_lin1(x))
        x = self.dropout(x)
        x = self.mlp_lin2(x)
        
        return x

if __name__ == "__main__":
    # Strict Sanity Check
    net = DynamicCWNetwork(in_channels=4, hidden_channels=32, out_classes=2)
    # Graph 1: 3 nodes, 2 edges
    d1 = Data(x=torch.randn(3, 4), edge_index=torch.tensor([[0, 1], [1, 2]]), edge_attr=torch.randn(2))
    # Graph 2: 4 nodes, 3 edges
    d2 = Data(x=torch.randn(4, 4), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]), edge_attr=torch.randn(3))
    # Batch them together
    batch_data = Batch.from_data_list([d1, d2])
    # Execute forward pass
    output = net(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
    # Prove tensor dimensions do not collapse improperly
    print(f"Batched Nodes Input Shape: {batch_data.x.shape}")
    print(f"Final Graph Output Shape: {output.shape}")
    assert output.shape == (2, 2), "Dimensionality collapsed improperly!"
    print("Sanity check passed!")
