import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import torch_geometric.utils as utils

from dynamic_cw_network import DynamicCWNetwork

# ==========================================
# 1. Data Preparation & Transform
# ==========================================
def compute_forman_ricci(data):
    """
    Computes the static combinatorial Forman-Ricci curvature:
    4 - deg(u) - deg(v) + 3 * num_triangles(e)
    Appends this as edge_attr.
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    # Calculate degree of each node
    deg = utils.degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    
    # Dense adjacency matrix for fast triangle counting
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1.0
    
    # adj_squared[u, v] gives the number of common neighbors (triangles for edge u,v)
    adj_squared = torch.matmul(adj, adj)
    
    u = edge_index[0]
    v = edge_index[1]
    num_triangles = adj_squared[u, v]
    
    # Calculate static Forman-Ricci curvature
    curvature = 4.0 - deg[u] - deg[v] + 3.0 * num_triangles
    
    # Store as edge_attr (requires shape [num_edges, 1])
    data.edge_attr = curvature.view(-1, 1)
    return data

class FormanRicciTransform(object):
    def __call__(self, data):
        return compute_forman_ricci(data)

# ==========================================
# 2. The Baseline Architecture (GINBaseline)
# ==========================================
class GINBaseline(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_classes):
        super(GINBaseline, self).__init__()
        
        # Initial Continuous Embedding
        self.initial_embedding = nn.Linear(in_channels, hidden_channels)
        
        # 3 GINConv layers with 2-layer MLP as standard
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))
        
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))
        
        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))
        
        # Classification Head
        self.mlp_lin1 = nn.Linear(hidden_channels, 16)
        self.dropout = nn.Dropout(p=0.5)
        self.mlp_lin2 = nn.Linear(16, out_classes)
        
    def forward(self, x, edge_index, batch):
        x = F.elu(self.initial_embedding(x))
        
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        
        # Global Readout Pooling
        x = global_add_pool(x, batch)
        
        # Classification Head
        x = F.elu(self.mlp_lin1(x))
        x = self.dropout(x)
        x = self.mlp_lin2(x)
        
        return x

# ==========================================
# 3. The Training Crucible
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading MUTAG dataset and calculating Forman-Ricci curvature...")
    dataset = TUDataset(root='./data', name='MUTAG', transform=FormanRicciTransform())
    
    # Shuffle and perform strict 80/20 train/test split (no data leakage)
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    
    split_idx = int(len(dataset) * 0.8)
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    in_channels = dataset.num_node_features
    out_classes = dataset.num_classes
    hidden_channels = 32
    
    # Instantiate models
    gin_model = GINBaseline(in_channels, hidden_channels, out_classes).to(device)
    cw_model = DynamicCWNetwork(in_channels, hidden_channels, out_classes).to(device)
    
    # Print parameter count check
    gin_params = sum(p.numel() for p in gin_model.parameters() if p.requires_grad)
    cw_params = sum(p.numel() for p in cw_model.parameters() if p.requires_grad)
    print(f"GINBaseline Parameters: {gin_params}")
    print(f"DynamicCWNetwork Parameters: {cw_params}")
    
    # Loss and Optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer_gin = torch.optim.Adam(gin_model.parameters(), lr=0.001)
    optimizer_cw = torch.optim.Adam(cw_model.parameters(), lr=0.001)
    
    best_gin_acc = 0.0
    best_cw_acc = 0.0
    
    print("\nStarting Training Crucible for 200 Epochs...")
    
    for epoch in range(1, 201):
        # --- Training Loop ---
        gin_model.train()
        cw_model.train()
        
        total_gin_loss = 0.0
        total_cw_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            # GINBaseline Forward & Update
            optimizer_gin.zero_grad()
            out_gin = gin_model(batch.x, batch.edge_index, batch.batch)
            loss_gin = criterion(out_gin, batch.y)
            loss_gin.backward()
            optimizer_gin.step()
            total_gin_loss += loss_gin.item() * batch.num_graphs
            
            # DynamicCWNetwork Forward & Update
            optimizer_cw.zero_grad()
            out_cw = cw_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss_cw = criterion(out_cw, batch.y)
            loss_cw.backward()
            optimizer_cw.step()
            total_cw_loss += loss_cw.item() * batch.num_graphs
            
        avg_gin_train_loss = total_gin_loss / len(train_loader.dataset)
        avg_cw_train_loss = total_cw_loss / len(train_loader.dataset)
        
        # --- Evaluation Loop ---
        gin_model.eval()
        cw_model.eval()
        
        gin_correct = 0
        cw_correct = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                
                # Evaluate GINBaseline
                out_gin = gin_model(batch.x, batch.edge_index, batch.batch)
                pred_gin = out_gin.argmax(dim=1)
                gin_correct += int((pred_gin == batch.y).sum())
                
                # Evaluate DynamicCWNetwork
                out_cw = cw_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                pred_cw = out_cw.argmax(dim=1)
                cw_correct += int((pred_cw == batch.y).sum())
                
        gin_acc = (gin_correct / len(test_loader.dataset)) * 100.0
        cw_acc = (cw_correct / len(test_loader.dataset)) * 100.0
        
        # Track best performance
        best_gin_acc = max(best_gin_acc, gin_acc)
        best_cw_acc = max(best_cw_acc, cw_acc)
        
        # Clean terminal output formatting
        print(f"Epoch [{epoch}/200] | GIN Train Loss: {avg_gin_train_loss:.2f}, GIN Test Acc: {gin_acc:.1f}% | CW-Net Train Loss: {avg_cw_train_loss:.2f}, CW-Net Test Acc: {cw_acc:.1f}%")
        
    # ==========================================
    # 4. Final Output and Winner Declaration
    # ==========================================
    print("\n" + "=" * 60)
    print("FINAL BENCHMARK RESULTS (Absolute Best Test Accuracy)")
    print("=" * 60)
    print(f"GINBaseline (Baseline)     : {best_gin_acc:.1f}%")
    print(f"DynamicCWNetwork (Custom)  : {best_cw_acc:.1f}%")
    
    if best_cw_acc > best_gin_acc:
        print("\nWINNER: DynamicCWNetwork! Your custom topology prevailed.")
    elif best_gin_acc > best_cw_acc:
        print("\nWINNER: GINBaseline! The standard message passing won this round.")
    else:
        print("\nWINNER: Tie! Both architectures reached the same peak performance.")

if __name__ == "__main__":
    main()
