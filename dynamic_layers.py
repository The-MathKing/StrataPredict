import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class LatentDynamicSimplicialConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # We use aggr="add" to sum the messages
        super(LatentDynamicSimplicialConv, self).__init__(aggr='add')
        
        self.lin_feat = nn.Linear(in_channels, out_channels)
        
        # Learnable beta for exponential decay, initialized to 1.0
        self.beta = nn.Parameter(torch.tensor(1.0))
        
        # Learnable projection vector for the scalar curvature
        self.w_curve = nn.Parameter(torch.Tensor(out_channels))
        
        # Initialize w_curve properly
        nn.init.xavier_uniform_(self.w_curve.unsqueeze(0))
        
    def forward(self, x, edge_index, edge_attr):
        """
        x: Node feature matrix of shape [num_nodes, in_channels]
        edge_index: Graph connectivity of shape [2, num_edges]
        edge_attr: Static topology metrics of shape [num_edges] or [num_edges, 1]
                   Expected value: 4 - deg(u) - deg(v) + 3 * num_triangles(e)
        """
        # Transform node features to out_channels
        x = self.lin_feat(x)
        
        # Ensure edge_attr is 2D for broadcasting
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
            
        # Start message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Add residual skip connection to prevent center node erasure
        out = out + x
        
        # Apply non-linear activation to prevent linear degradation
        return F.elu(out)
        
    def message(self, x_i, x_j, edge_attr):
        """
        x_i: Target node features [num_edges, out_channels]
        x_j: Source node features [num_edges, out_channels]
        edge_attr: Static topological term [num_edges, 1]
        """
        # 1. Compute squared Euclidean distance ||h_u - h_v||_2^2
        dist_sq = torch.sum((x_i - x_j) ** 2, dim=-1, keepdim=True)
        
        # 2. Compute dynamic exponential decay with Softplus constraint to prevent gradient explosion
        decay = torch.exp(-F.softplus(self.beta) * dist_sq)
        
        # 3. Compute dynamic curvature scalar
        f_dynamic = edge_attr * decay
        
        # 4. Project the scalar into the feature space using the learnable w_curve
        # Output Message: W_feat * x_j + (F_dynamic * w_curve)
        # Note: x_j is already transformed by W_feat via self.lin_feat in forward()
        message = x_j + (f_dynamic * self.w_curve)
        
        return message
