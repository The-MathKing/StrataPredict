import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import KFold
import numpy as np
import time

from train import process_dataset

def safe_mean(tensor, batch):
    """Safely calculates global mean pool without throwing NaNs on empty tensors."""
    if tensor is None or tensor.size(0) == 0:
        return torch.zeros(1, tensor.size(1) if tensor.dim() > 1 else 1, device=tensor.device)
    if batch is None:
        return torch.mean(tensor, dim=0, keepdim=True)
    
    num_graphs = int(batch.max().item() + 1)
    out = torch.zeros(num_graphs, tensor.size(1), device=tensor.device)
    count = torch.zeros(num_graphs, 1, device=tensor.device)
    
    out.scatter_add_(0, batch.unsqueeze(-1).expand_as(tensor), tensor)
    count.scatter_add_(0, batch.unsqueeze(-1), torch.ones_like(batch.unsqueeze(-1), dtype=torch.float))
    
    return out / count.clamp(min=1)

class AblationCurvatureConv(nn.Module):
    def __init__(self, in_channels, out_channels, ablation_mode="gating"):
        super(AblationCurvatureConv, self).__init__()
        self.ablation_mode = ablation_mode
        
        # We adjust input channels for concatenation mode since it adds 1 feature
        self.edge_in_channels = in_channels + 1 if ablation_mode == "concat" else in_channels
        
        self.lin_down = nn.Linear(in_channels, out_channels)
        self.lin_up = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(self.edge_in_channels, out_channels)
        self.lin_adj = nn.Linear(self.edge_in_channels, out_channels)
        
        if ablation_mode == "addition":
            self.bias_weight = nn.Parameter(torch.Tensor(1, out_channels))
            nn.init.xavier_uniform_(self.bias_weight)
            
    def forward(self, x_0, x_1, x_2, incidence_1, incidence_2, frc_weights):
        # 0. Feature modification for concat mode
        if self.ablation_mode == "concat":
            # Append FRC to the raw features
            x_1_input = torch.cat([x_1, frc_weights], dim=-1)
        else:
            x_1_input = x_1
            
        x_1_t = self.lin_self(x_1_input)
        
        if x_0 is not None:
            x_0_t = self.lin_up(x_0)
            
        if x_2 is not None:
            x_2_t = self.lin_down(x_2)
            
        x_1_adj = self.lin_adj(x_1_input)
            
        # Message passing
        msg_up = torch.sparse.mm(incidence_1.t(), x_0_t) if x_0 is not None else 0
        msg_down = torch.sparse.mm(incidence_2, x_2_t) if x_2 is not None else 0
        
        msg_adj_down = torch.sparse.mm(incidence_1.t(), torch.sparse.mm(incidence_1, x_1_adj)) if incidence_1.shape[1] > 0 else 0
        msg_adj_up = torch.sparse.mm(incidence_2, torch.sparse.mm(incidence_2.t(), x_1_adj)) if incidence_2.shape[1] > 0 else 0
        
        # 1. Topological Dropout (Pruning)
        if self.ablation_mode == "dropout" and self.training:
            # Drop edges with negative FRC with p=0.5
            drop_mask = (frc_weights < 0).float()
            drop_prob = torch.rand_like(drop_mask)
            keep_mask = 1.0 - (drop_mask * (drop_prob < 0.5).float())
            out = (x_1_t + msg_up + msg_down + msg_adj_down + msg_adj_up) * keep_mask
        else:
            out = x_1_t + msg_up + msg_down + msg_adj_down + msg_adj_up
        
        # Apply the specific Curvature Integration Method
        if self.ablation_mode == "gating":
            out = out * torch.sigmoid(frc_weights)
        elif self.ablation_mode == "addition":
            out = out + (frc_weights * self.bias_weight)
        elif self.ablation_mode == "inverse_penalty":
            out = out * torch.sigmoid(-frc_weights)
        # concat and dropout are handled earlier
            
        return F.elu(out)

class AblationMPSN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, ablation_mode):
        super(AblationMPSN, self).__init__()
        
        self.node_emb = nn.Linear(num_node_features, hidden_dim)
        self.edge_emb = nn.Linear(2, hidden_dim) # MUTAG has 2 edge features usually, or 1 if empty
        self.tri_emb = nn.Linear(1, hidden_dim)
        
        self.conv1 = AblationCurvatureConv(hidden_dim, hidden_dim, ablation_mode)
        self.conv2 = AblationCurvatureConv(hidden_dim, hidden_dim, ablation_mode)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x_0, x_1, x_2, B1, B2, frc, batch_0, batch_1, batch_2):
        x_0 = self.node_emb(x_0)
        
        if x_1 is None or x_1.size(0) == 0:
            x_1 = torch.ones((B1.shape[1] if B1.shape[1]>0 else 0, self.edge_emb.in_features), device=x_0.device)
        x_1 = self.edge_emb(x_1)
        
        if x_2 is None or x_2.size(0) == 0:
            x_2 = torch.ones((B2.shape[1] if B2.shape[1]>0 else 0, 1), device=x_0.device)
        x_2 = self.tri_emb(x_2)
        
        # Message passing layers
        x_1_out = self.conv1(x_0, x_1, x_2, B1, B2, frc)
        x_1_out = self.conv2(x_0, x_1_out, x_2, B1, B2, frc)
        
        # Pooling
        pooled_0 = safe_mean(x_0, batch_0)
        pooled_1 = safe_mean(x_1_out, batch_1)
        pooled_2 = safe_mean(x_2, batch_2)
        
        final_embedding = torch.cat([pooled_0, pooled_1, pooled_2], dim=1)
        out = self.classifier(final_embedding)
        return out

def run_ablation_study():
    print("Loading MUTAG dataset for fast ablation study...")
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    processed_dataset = process_dataset(dataset)
    
    num_node_features = dataset.num_node_features if dataset.num_node_features > 0 else 1
    num_classes = dataset.num_classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    modes = ["gating", "concat", "addition", "inverse_penalty", "dropout"]
    results = {}
    
    print("\nStarting 5-Fold Cross Validation for each mode...")
    
    for mode in modes:
        print(f"\n==============================")
        print(f"Testing Mode: {mode.upper()}")
        print(f"==============================")
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_accs = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(processed_dataset)):
            train_data = [processed_dataset[i] for i in train_idx]
            test_data = [processed_dataset[i] for i in test_idx]
            
            model = AblationMPSN(num_node_features, 32, num_classes, ablation_mode=mode).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            
            best_test = 0.0
            
            # Train 20 epochs
            for epoch in range(1, 21):
                model.train()
                for data in train_data:
                    optimizer.zero_grad()
                    out = model(data['x_0'].to(device), None, None, 
                                data['B1'].to(device), data['B2'].to(device), 
                                data['frc'].to(device), None, None, None)
                    loss = criterion(out, data['y'].to(device))
                    loss.backward()
                    optimizer.step()
                
                # Test
                model.eval()
                correct = 0
                with torch.no_grad():
                    for data in test_data:
                        out = model(data['x_0'].to(device), None, None, 
                                    data['B1'].to(device), data['B2'].to(device), 
                                    data['frc'].to(device), None, None, None)
                        pred = out.argmax(dim=1)
                        correct += int((pred == data['y'].to(device)).sum())
                        
                acc = correct / len(test_data)
                if acc > best_test:
                    best_test = acc
                    
            all_accs.append(best_test)
            print(f"Fold {fold+1} Best Acc: {best_test*100:.2f}%")
            
        mean_acc = np.mean(all_accs)
        std_acc = np.std(all_accs)
        results[mode] = (mean_acc, std_acc)
        print(f"-> {mode.upper()} Final Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

    print("\n\n####################################")
    print("FINAL ABLATION STUDY RESULTS")
    print("####################################")
    for mode, (mean, std) in results.items():
        print(f"{mode.upper():<20}: {mean*100:.2f}% ± {std*100:.2f}%")

if __name__ == "__main__":
    run_ablation_study()
