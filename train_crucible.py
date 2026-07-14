import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import torch_geometric.utils as utils
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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
# 3. The Rigorous Training Crucible
# ==========================================
def main():
    # Due to dynamic graph batch shapes, PyTorch MPS causes a shader cache explosion.
    # Apple Silicon CPU is significantly faster for PyTorch Geometric workloads because
    # it doesn't have to recompile shaders for every batch size variation.
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    seeds = [42, 123, 2026, 777, 999]
    num_epochs = 200
    
    # Arrays to store telemetry for plotting [seed, epoch]
    all_gin_train_losses = np.zeros((len(seeds), num_epochs))
    all_gin_test_accs = np.zeros((len(seeds), num_epochs))
    all_cw_train_losses = np.zeros((len(seeds), num_epochs))
    all_cw_test_accs = np.zeros((len(seeds), num_epochs))
    
    # Dictionaries to store final statistical metrics
    final_metrics = {
        'GIN': {'acc': [], 'f1': [], 'runtime': [], 'cm': []},
        'CW': {'acc': [], 'f1': [], 'runtime': [], 'cm': []}
    }
    
    print("Loading NCI1 dataset and calculating Forman-Ricci curvature...")
    base_dataset = TUDataset(root='./data', name='NCI1', pre_transform=FormanRicciTransform())
    
    in_channels = base_dataset.num_node_features
    out_classes = base_dataset.num_classes
    hidden_channels = 32

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"ANTI-FLUKE PROTOCOL: Starting Seed {seed} ({seed_idx + 1}/{len(seeds)})")
        print(f"{'='*60}")
        
        # Reset seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # Shuffle dataset cleanly for this seed
        dataset = base_dataset.shuffle()
        
        # 80/20 Train/Test split
        split_idx = int(len(dataset) * 0.8)
        train_dataset = dataset[:split_idx]
        test_dataset = dataset[split_idx:]
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Instantiate fresh models for each seed
        gin_model = GINBaseline(in_channels, hidden_channels, out_classes).to(device)
        cw_model = DynamicCWNetwork(in_channels, hidden_channels, out_classes).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer_gin = torch.optim.Adam(gin_model.parameters(), lr=0.001)
        optimizer_cw = torch.optim.Adam(cw_model.parameters(), lr=0.001)
        
        gin_epoch_runtimes = []
        cw_epoch_runtimes = []
        
        for epoch in range(1, num_epochs + 1):
            # --- Training Loop ---
            gin_model.train()
            cw_model.train()
            
            total_gin_loss = 0.0
            total_cw_loss = 0.0
            
            # 1. GINBaseline Execution & Telemetry
            start_time = time.time()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer_gin.zero_grad()
                out_gin = gin_model(batch.x, batch.edge_index, batch.batch)
                loss_gin = criterion(out_gin, batch.y)
                loss_gin.backward()
                optimizer_gin.step()
                total_gin_loss += loss_gin.item() * batch.num_graphs
            gin_epoch_runtimes.append(time.time() - start_time)
            
            # 2. DynamicCWNetwork Execution & Telemetry
            start_time = time.time()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer_cw.zero_grad()
                out_cw = cw_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss_cw = criterion(out_cw, batch.y)
                loss_cw.backward()
                optimizer_cw.step()
                total_cw_loss += loss_cw.item() * batch.num_graphs
            cw_epoch_runtimes.append(time.time() - start_time)
            
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
                    
                    out_gin = gin_model(batch.x, batch.edge_index, batch.batch)
                    pred_gin = out_gin.argmax(dim=1)
                    gin_correct += int((pred_gin == batch.y).sum())
                    
                    out_cw = cw_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    pred_cw = out_cw.argmax(dim=1)
                    cw_correct += int((pred_cw == batch.y).sum())
                    
            gin_acc = (gin_correct / len(test_loader.dataset)) * 100.0
            cw_acc = (cw_correct / len(test_loader.dataset)) * 100.0
            
            # Store telemetry
            all_gin_train_losses[seed_idx, epoch - 1] = avg_gin_train_loss
            all_gin_test_accs[seed_idx, epoch - 1] = gin_acc
            all_cw_train_losses[seed_idx, epoch - 1] = avg_cw_train_loss
            all_cw_test_accs[seed_idx, epoch - 1] = cw_acc
            
            # Reduce print spam (Print every 20 epochs or at the end)
            if epoch % 20 == 0 or epoch == 1:
                print(f"Epoch [{epoch}/{num_epochs}] | "
                      f"GIN Train Loss: {avg_gin_train_loss:.2f}, Test Acc: {gin_acc:.1f}% | "
                      f"CW Train Loss: {avg_cw_train_loss:.2f}, Test Acc: {cw_acc:.1f}%")
        
        # --- End of 200 Epochs: Advanced Classification Metrics ---
        gin_preds, gin_targets = [], []
        cw_preds, cw_targets = [], []
        
        gin_model.eval()
        cw_model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                
                out_gin = gin_model(batch.x, batch.edge_index, batch.batch)
                out_cw = cw_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                gin_preds.extend(out_gin.argmax(dim=1).cpu().numpy())
                gin_targets.extend(batch.y.cpu().numpy())
                
                cw_preds.extend(out_cw.argmax(dim=1).cpu().numpy())
                cw_targets.extend(batch.y.cpu().numpy())
                
        # Scikit-learn validations (Labels=[0,1] forces a 2x2 confusion matrix)
        gin_f1 = f1_score(gin_targets, gin_preds, average='macro')
        gin_cm = confusion_matrix(gin_targets, gin_preds, labels=[0, 1])
        gin_acc_final = accuracy_score(gin_targets, gin_preds) * 100.0
        
        cw_f1 = f1_score(cw_targets, cw_preds, average='macro')
        cw_cm = confusion_matrix(cw_targets, cw_preds, labels=[0, 1])
        cw_acc_final = accuracy_score(cw_targets, cw_preds) * 100.0
        
        avg_gin_runtime = np.mean(gin_epoch_runtimes)
        avg_cw_runtime = np.mean(cw_epoch_runtimes)
        
        # Append to final stats arrays
        final_metrics['GIN']['acc'].append(gin_acc_final)
        final_metrics['GIN']['f1'].append(gin_f1)
        final_metrics['GIN']['runtime'].append(avg_gin_runtime)
        final_metrics['GIN']['cm'].append(gin_cm.tolist())
        
        final_metrics['CW']['acc'].append(cw_acc_final)
        final_metrics['CW']['f1'].append(cw_f1)
        final_metrics['CW']['runtime'].append(avg_cw_runtime)
        final_metrics['CW']['cm'].append(cw_cm.tolist())
        
        print(f"\n--- Seed {seed} Final Stats ---")
        print(f"GINBaseline  -> Acc: {gin_acc_final:.2f}%, F1 (Macro): {gin_f1:.4f}, Avg Runtime/Epoch: {avg_gin_runtime:.4f}s")
        print(f"DynamicCWNet -> Acc: {cw_acc_final:.2f}%, F1 (Macro): {cw_f1:.4f}, Avg Runtime/Epoch: {avg_cw_runtime:.4f}s")

    # ==========================================
    # 4. Presentation-Ready Data Export
    # ==========================================
    print("\n" + "=" * 60)
    print("CALCULATING STATISTICAL SIGNIFICANCE & EXPORTING TELEMETRY")
    print("=" * 60)
    
    # 1. Export epoch_loss_curves.csv
    epoch_df = pd.DataFrame({
        'Epoch': np.arange(1, num_epochs + 1),
        'GIN_Train_Loss_Mean': np.mean(all_gin_train_losses, axis=0),
        'GIN_Test_Acc_Mean': np.mean(all_gin_test_accs, axis=0),
        'CW_Train_Loss_Mean': np.mean(all_cw_train_losses, axis=0),
        'CW_Test_Acc_Mean': np.mean(all_cw_test_accs, axis=0)
    })
    epoch_df.to_csv('epoch_loss_curves.csv', index=False)
    print(">> Exported: epoch_loss_curves.csv (Plotting Data)")
    
    # 2. Export final_statistics.json
    def build_stats_dict(model_key):
        arr = final_metrics[model_key]
        # Average the 5 confusion matrices
        cm_avg = np.mean(arr['cm'], axis=0)
        # Extract TP, FP, TN, FN safely from a 2x2
        tn, fp, fn, tp = cm_avg.ravel()
        
        return {
            'Accuracy': {
                'Mean': float(np.mean(arr['acc'])),
                'Std_Dev': float(np.std(arr['acc']))
            },
            'F1_Macro': {
                'Mean': float(np.mean(arr['f1'])),
                'Std_Dev': float(np.std(arr['f1']))
            },
            'Average_Runtime_Per_Epoch_Sec': {
                'Mean': float(np.mean(arr['runtime'])),
                'Std_Dev': float(np.std(arr['runtime']))
            },
            'Confusion_Matrix_Averaged': {
                'True_Positives (TP)': float(tp),
                'True_Negatives (TN)': float(tn),
                'False_Positives (FP)': float(fp),
                'False_Negatives (FN)': float(fn)
            },
            'Confusion_Matrices_Raw': arr['cm']
        }

    final_stats_json = {
        'Statistical_Significance_Summary': "Mean ± Std Dev across 5 distinct random seeds.",
        'GINBaseline': build_stats_dict('GIN'),
        'DynamicCWNetwork': build_stats_dict('CW')
    }
    
    with open('final_statistics.json', 'w') as f:
        json.dump(final_stats_json, f, indent=4)
        
    print(">> Exported: final_statistics.json (Publication Data)")
    print("\nRigorous evaluation suite completed successfully.")

if __name__ == "__main__":
    main()
