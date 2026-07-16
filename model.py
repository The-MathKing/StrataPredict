import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GINCellularMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GINCellularMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )
        self.eps = nn.Parameter(torch.rand(1) * 0.5 + 0.1)
        
    def forward(self, x, messages):
        if x is None:
            return self.net(messages)
        return self.net((1 + self.eps) * x + messages)

class CurvatureWeightedSimplicialConv(nn.Module):
    def __init__(self, in_channels, out_channels, gating='curvature'):
        super(CurvatureWeightedSimplicialConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gating = gating
        
        # GIN Cellular Aggregations
        self.mlp_edge = GINCellularMLP(in_channels, out_channels)
        self.mlp_node = GINCellularMLP(in_channels, out_channels)
        self.mlp_tri = GINCellularMLP(in_channels, out_channels)
        
        # Linear transformations for the different message passing channels
        self.lin_down = nn.Linear(in_channels, in_channels)
        self.lin_up = nn.Linear(in_channels, in_channels)
        self.lin_adj = nn.Linear(in_channels, in_channels)
        
        # Node updates
        self.lin_node_edge = nn.Linear(in_channels, in_channels)
        
        # Triangle updates
        self.lin_tri_edge = nn.Linear(in_channels, in_channels)
        
        if self.gating == 'scalar':
            self.scalar_gate = nn.Linear(in_channels, 1)
        elif self.gating == 'vector':
            self.vector_gate = nn.Linear(in_channels, out_channels)
            
        # Dynamic Ricci Rewiring parameter
        self.tau = nn.Parameter(torch.tensor(0.0))
        
        # For rewiring soft attention
        self.q_proj = nn.Linear(out_channels, out_channels)
        self.k_proj = nn.Linear(out_channels, out_channels)
        self.v_proj = nn.Linear(out_channels, out_channels)
        
    def dynamic_ricci_rewiring(self, x_0, incidence_1, frc_weights):
        # Identify severed edges
        M_sever = (torch.abs(frc_weights) < self.tau).squeeze(1).float()
        
        if M_sever.sum() == 0 or x_0 is None:
            return torch.zeros_like(x_0) if x_0 is not None else 0
            
        # Compute A_sever = |B_1| diag(M_sever) |B_1|^T
        if incidence_1.is_sparse:
            inc_1_c = incidence_1.coalesce()
            abs_B1 = torch.sparse_coo_tensor(inc_1_c.indices(), torch.abs(inc_1_c.values()), incidence_1.shape)
        else:
            abs_B1 = torch.abs(incidence_1)
            
        abs_B1_dense = abs_B1.to_dense() if abs_B1.is_sparse else abs_B1
        
        A_sever = torch.matmul(abs_B1_dense * M_sever.unsqueeze(0), abs_B1_dense.t())
        A_orig = torch.matmul(abs_B1_dense, abs_B1_dense.t())
        
        A_rewire = torch.matmul(A_orig, torch.matmul(A_sever, A_orig))
        
        # Soft attention over non-zero entries of A_rewire
        mask = (A_rewire > 0).float()
        # Remove self loops from mask
        mask.fill_diagonal_(0)
        
        if mask.sum() == 0:
            return torch.zeros_like(x_0)
            
        Q = self.q_proj(x_0)
        K = self.k_proj(x_0)
        V = self.v_proj(x_0)
        
        scores = torch.matmul(Q, K.t()) / math.sqrt(self.out_channels)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        
        attn_out = torch.matmul(attn, V)
        return attn_out

    def forward(self, x_0, x_1, x_2, incidence_1, incidence_2, frc_weights):
        # 3. CurvatureLayerNorm
        if frc_weights.shape[0] > 1:
            frc_mean = frc_weights.mean()
            frc_std = frc_weights.std()
            frc_norm = (frc_weights - frc_mean) / (frc_std + 1e-5)
        else:
            frc_norm = frc_weights
            
        # Edge aggregation
        if x_0 is not None:
            x_0_t = self.lin_up(x_0)
            msg_up = torch.sparse.mm(incidence_1.t(), x_0_t) if incidence_1.is_sparse else torch.matmul(incidence_1.t(), x_0_t)
        else:
            msg_up = 0
            
        if x_2 is not None:
            x_2_t = self.lin_down(x_2)
            msg_down = torch.sparse.mm(incidence_2, x_2_t) if incidence_2.is_sparse else torch.matmul(incidence_2, x_2_t)
        else:
            msg_down = 0
            
        x_1_adj = self.lin_adj(x_1)
        if incidence_1.shape[1] > 0:
            if incidence_1.is_sparse:
                msg_adj_down = torch.sparse.mm(incidence_1.t(), torch.sparse.mm(incidence_1, x_1_adj))
            else:
                msg_adj_down = torch.matmul(incidence_1.t(), torch.matmul(incidence_1, x_1_adj))
        else:
            msg_adj_down = 0
            
        if incidence_2.shape[1] > 0:
            if incidence_2.is_sparse:
                msg_adj_up = torch.sparse.mm(incidence_2, torch.sparse.mm(incidence_2.t(), x_1_adj))
            else:
                msg_adj_up = torch.matmul(incidence_2, torch.matmul(incidence_2.t(), x_1_adj))
        else:
            msg_adj_up = 0
            
        msg_edge = msg_up + msg_down + msg_adj_down + msg_adj_up
        
        out_edge = self.mlp_edge(x_1, msg_edge)
        
        # Apply Gating with Normalized Curvature
        if self.gating == 'curvature':
            gate = torch.sigmoid(frc_norm)
            out_edge = out_edge * gate
        elif self.gating == 'scalar':
            gate = torch.sigmoid(self.scalar_gate(x_1))
            out_edge = out_edge * gate
        elif self.gating == 'vector':
            gate = torch.sigmoid(self.vector_gate(x_1))
            out_edge = out_edge * gate
            
        x_1_new = F.gelu(out_edge)
        
        # Node update
        if x_0 is not None:
            if incidence_1.shape[1] == 0:
                msg_edge_to_node = torch.zeros((incidence_1.shape[0], x_1.shape[1]), device=x_1.device)
            elif incidence_1.is_sparse:
                inc_1_c = incidence_1.coalesce()
                abs_inc_1 = torch.sparse_coo_tensor(inc_1_c.indices(), torch.abs(inc_1_c.values()), incidence_1.shape)
                msg_edge_to_node = torch.sparse.mm(abs_inc_1, x_1)
            else:
                abs_inc_1 = torch.abs(incidence_1)
                msg_edge_to_node = torch.matmul(abs_inc_1, x_1)
                
            msg_node = self.lin_node_edge(msg_edge_to_node)
            
            # Dynamic Ricci Rewiring (soft attention)
            attn_mass = self.dynamic_ricci_rewiring(x_0, incidence_1, frc_norm)
            msg_node = msg_node + attn_mass
            
            x_0_new = self.mlp_node(x_0, msg_node)
            x_0_new = F.gelu(x_0_new)
        else:
            x_0_new = None
            
        # Triangle update
        if x_2 is not None:
            if incidence_2.shape[0] == 0:
                msg_edge_to_tri = torch.zeros((incidence_2.shape[1], x_1.shape[1]), device=x_1.device)
            elif incidence_2.is_sparse:
                inc_2_c = incidence_2.coalesce()
                abs_inc_2 = torch.sparse_coo_tensor(inc_2_c.indices(), torch.abs(inc_2_c.values()), incidence_2.shape)
                msg_edge_to_tri = torch.sparse.mm(abs_inc_2.t(), x_1)
            else:
                abs_inc_2 = torch.abs(incidence_2)
                msg_edge_to_tri = torch.matmul(abs_inc_2.t(), x_1)
                
            msg_tri = self.lin_tri_edge(msg_edge_to_tri)
            x_2_new = self.mlp_tri(x_2, msg_tri)
            x_2_new = F.gelu(x_2_new)
        else:
            x_2_new = None
            
        return x_0_new, x_1_new, x_2_new

class CurvatureMPSN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, gating='curvature'):
        super(CurvatureMPSN, self).__init__()
        
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        self.edge_embedding = nn.Linear(1, hidden_dim) 
        self.triangle_embedding = nn.Linear(1, hidden_dim)
        
        self.conv1 = CurvatureWeightedSimplicialConv(hidden_dim, hidden_dim, gating=gating)
        self.conv2 = CurvatureWeightedSimplicialConv(hidden_dim, hidden_dim, gating=gating)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), 
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x_0, x_1, x_2, incidence_1, incidence_2, frc_weights, batch_0, batch_1, batch_2):
        x_0 = self.node_embedding(x_0)
        
        if x_1 is None:
            x_1 = torch.ones((incidence_1.shape[1], 1), device=x_0.device)
        x_1 = self.edge_embedding(x_1)
        
        if x_2 is None:
            x_2 = torch.ones((incidence_2.shape[1], 1), device=x_0.device)
        x_2 = self.triangle_embedding(x_2)
        
        x_0, x_1, x_2 = self.conv1(x_0, x_1, x_2, incidence_1, incidence_2, frc_weights)
        x_0, x_1, x_2 = self.conv2(x_0, x_1, x_2, incidence_1, incidence_2, frc_weights)
        
        from torch_geometric.nn import global_mean_pool
        
        def safe_mean(x, batch):
            if x.shape[0] == 0:
                return torch.zeros((1, x.shape[1]), device=x.device)
            if batch is not None:
                return global_mean_pool(x, batch)
            return torch.mean(x, dim=0, keepdim=True)
        
        pooled_0 = safe_mean(x_0, batch_0)
        pooled_1 = safe_mean(x_1, batch_1)
        pooled_2 = safe_mean(x_2, batch_2)
        
        graph_embedding = torch.cat([pooled_0, pooled_1, pooled_2], dim=1)
        
        out = self.classifier(graph_embedding)
        return out
