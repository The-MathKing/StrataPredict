import torch
import torch.nn as nn
import torch.nn.functional as F
from toponetx.classes import SimplicialComplex

class CurvatureWeightedSimplicialConv(nn.Module):
    """
    A custom Topological Message Passing Layer that weights edge communications
    by their Discrete Forman-Ricci Curvature.
    
    This layer specifically operates on 1-cells (edges) of a Simplicial Complex.
    """
    def __init__(self, in_channels, out_channels, gating='curvature'):
        super(CurvatureWeightedSimplicialConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gating = gating
        
        # Linear transformations for the different message passing channels
        # Downward message passing (from triangles to edges)
        self.lin_down = nn.Linear(in_channels, out_channels)
        # Upward message passing (from nodes to edges)
        self.lin_up = nn.Linear(in_channels, out_channels)
        # Self-update (edge to edge)
        self.lin_self = nn.Linear(in_channels, out_channels)
        # Adjacency message passing (Hodge Laplacian)
        self.lin_adj = nn.Linear(in_channels, out_channels)
        
        if self.gating == 'scalar':
            self.scalar_gate = nn.Linear(in_channels, 1)
        elif self.gating == 'vector':
            self.vector_gate = nn.Linear(in_channels, out_channels)
            
    def forward(self, x_0, x_1, x_2, incidence_1, incidence_2, frc_weights):
        """
        Forward pass for the curvature-weighted convolutional layer.
        
        Parameters:
        x_0: Node features (0-cells)
        x_1: Edge features (1-cells)
        x_2: Triangle features (2-cells)
        incidence_1: Boundary matrix B_1 (nodes x edges)
        incidence_2: Boundary matrix B_2 (edges x triangles)
        frc_weights: The Forman-Ricci Curvature tensor for edges
        """
        # Apply non-linear transformations to features
        if x_0 is not None:
            x_0_t = self.lin_up(x_0)
        
        x_1_t = self.lin_self(x_1)
        
        if x_2 is not None:
            x_2_t = self.lin_down(x_2)
            
        x_1_adj = self.lin_adj(x_1)
            
        # Message passing using incidence matrices
        
        # 1. Upward from nodes to edges: B_1^T @ x_0
        # Incidence_1 is size (N_nodes, N_edges). Transpose to map nodes -> edges
        msg_up = torch.sparse.mm(incidence_1.t(), x_0_t) if x_0 is not None else 0
        
        # 2. Downward from triangles to edges: B_2 @ x_2
        # Incidence_2 is size (N_edges, N_triangles). Maps triangles -> edges
        msg_down = torch.sparse.mm(incidence_2, x_2_t) if x_2 is not None else 0
        
        # 3. Adjacency via Hodge Laplacian: (B_1^T B_1 + B_2 B_2^T) @ x_1
        # Edge to edge via nodes
        msg_adj_down = torch.sparse.mm(incidence_1.t(), torch.sparse.mm(incidence_1, x_1_adj)) if incidence_1.shape[1] > 0 else 0
        # Edge to edge via triangles
        msg_adj_up = torch.sparse.mm(incidence_2, torch.sparse.mm(incidence_2.t(), x_1_adj)) if incidence_2.shape[1] > 0 else 0
        
        # Aggregate messages
        out = x_1_t + msg_up + msg_down + msg_adj_down + msg_adj_up
        
        # Apply Gating
        if self.gating == 'curvature':
            gate = torch.sigmoid(frc_weights)
            out = out * gate
        elif self.gating == 'scalar':
            gate = torch.sigmoid(self.scalar_gate(x_1))
            out = out * gate
        elif self.gating == 'vector':
            gate = torch.sigmoid(self.vector_gate(x_1))
            out = out * gate
            
        return F.elu(out)


class CurvatureMPSN(nn.Module):
    """
    Curvature-Weighted Message Passing Simplicial Network
    
    A full architecture containing topological layers and a global readout.
    """
    def __init__(self, num_node_features, hidden_dim, num_classes, gating='curvature'):
        super(CurvatureMPSN, self).__init__()
        
        # Initial embeddings
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        # We'll initialize edge and triangle features if they don't exist
        self.edge_embedding = nn.Linear(1, hidden_dim) # e.g. initialize with edge lengths or random
        self.triangle_embedding = nn.Linear(1, hidden_dim)
        
        # Convolutional layers
        self.conv1 = CurvatureWeightedSimplicialConv(hidden_dim, hidden_dim, gating=gating)
        self.conv2 = CurvatureWeightedSimplicialConv(hidden_dim, hidden_dim, gating=gating)
        
        # Final classification readout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), # *3 because we concatenate pooled nodes, edges, triangles
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x_0, x_1, x_2, incidence_1, incidence_2, frc_weights, batch_0, batch_1, batch_2):
        # Initial projection
        x_0 = self.node_embedding(x_0)
        
        # Usually datasets don't have native edge/triangle features, 
        # we can initialize them as ones and project
        if x_1 is None:
            x_1 = torch.ones((incidence_1.shape[1], 1), device=x_0.device)
        x_1 = self.edge_embedding(x_1)
        
        if x_2 is None:
            x_2 = torch.ones((incidence_2.shape[1], 1), device=x_0.device)
        x_2 = self.triangle_embedding(x_2)
        
        # Topological Message Passing
        # In a full model, we'd also have node and triangle update layers,
        # but for this experiment, the novelty is the curvature weighted edge update.
        x_1 = self.conv1(x_0, x_1, x_2, incidence_1, incidence_2, frc_weights)
        x_1 = self.conv2(x_0, x_1, x_2, incidence_1, incidence_2, frc_weights)
        
        # Global Pooling (Readout)
        from torch_geometric.nn import global_mean_pool
        
        def safe_mean(x, batch):
            if x.shape[0] == 0:
                return torch.zeros((1, x.shape[1]), device=x.device)
            if batch is not None:
                return global_mean_pool(x, batch)
            return torch.mean(x, dim=0, keepdim=True)
        
        # Pool all nodes, edges, triangles in the graph safely
        pooled_0 = safe_mean(x_0, batch_0)
        pooled_1 = safe_mean(x_1, batch_1)
        pooled_2 = safe_mean(x_2, batch_2)
        
        # Concatenate topological features
        graph_embedding = torch.cat([pooled_0, pooled_1, pooled_2], dim=1)
        
        # Classification
        out = self.classifier(graph_embedding)
        return out
