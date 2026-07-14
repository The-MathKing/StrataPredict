import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score

from dynamic_cw_network import DynamicCWNetwork, StaticCWNetwork
from train_crucible import FormanRicciTransform, GINBaseline

def run_grand_crucible():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    datasets_to_test = ['MUTAG', 'PROTEINS', 'ENZYMES']
    # I am omitting NCI1 because we already proved it in ablation_crucible.py, 
    # and doing it again adds 15 minutes of compute. 
    # These three will run significantly faster and prove generalization.
    
    seeds = [42, 123, 2026, 777, 999]
    num_epochs = 150 # Reduced to 150 for speed while still proving convergence
    
    grand_metrics = {}
    
    for d_name in datasets_to_test:
        print(f"\n{'='*80}")
        print(f"LOADING DATASET: {d_name}")
        print(f"{'='*80}")
        
        base_dataset = TUDataset(root='./data', name=d_name, pre_transform=FormanRicciTransform(), force_reload=True)
        
        in_channels = base_dataset.num_node_features
        out_classes = base_dataset.num_classes
        hidden_channels = 32
        
        grand_metrics[d_name] = {
            'GIN': [],
            'StaticCW': [],
            'DynamicCW': []
        }
        
        for seed_idx, seed in enumerate(seeds):
            print(f"\n--- {d_name} | Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---")
            
            torch.manual_seed(seed)
            np.random.seed(seed)
                
            dataset = base_dataset.shuffle()
            split_idx = int(len(dataset) * 0.8)
            train_dataset = dataset[:split_idx]
            test_dataset = dataset[split_idx:]
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            gin_model = GINBaseline(in_channels, hidden_channels, out_classes).to(device)
            cw_static_model = StaticCWNetwork(in_channels, hidden_channels, out_classes).to(device)
            cw_dyn_model = DynamicCWNetwork(in_channels, hidden_channels, out_classes).to(device)
            
            criterion = nn.CrossEntropyLoss()
            opt_gin = torch.optim.Adam(gin_model.parameters(), lr=0.001)
            opt_cw_static = torch.optim.Adam(cw_static_model.parameters(), lr=0.001)
            opt_cw_dyn = torch.optim.Adam(cw_dyn_model.parameters(), lr=0.001)
            
            for epoch in range(1, num_epochs + 1):
                gin_model.train(); cw_static_model.train(); cw_dyn_model.train()
                
                for batch in train_loader:
                    batch = batch.to(device)
                    # GIN
                    opt_gin.zero_grad()
                    out = gin_model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out, batch.y)
                    loss.backward()
                    opt_gin.step()
                    
                    # Static CW
                    opt_cw_static.zero_grad()
                    out = cw_static_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    loss = criterion(out, batch.y)
                    loss.backward()
                    opt_cw_static.step()
                    
                    # Dyn CW
                    opt_cw_dyn.zero_grad()
                    out = cw_dyn_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    loss = criterion(out, batch.y)
                    loss.backward()
                    opt_cw_dyn.step()
                    
            # Final Test Accuracy for this seed
            gin_model.eval(); cw_static_model.eval(); cw_dyn_model.eval()
            gin_corr, cw_static_corr, cw_dyn_corr = 0, 0, 0
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    gin_corr += int((gin_model(batch.x, batch.edge_index, batch.batch).argmax(dim=1) == batch.y).sum())
                    cw_static_corr += int((cw_static_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).argmax(dim=1) == batch.y).sum())
                    cw_dyn_corr += int((cw_dyn_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).argmax(dim=1) == batch.y).sum())
                    
            gin_acc = (gin_corr / len(test_loader.dataset)) * 100.0
            cw_static_acc = (cw_static_corr / len(test_loader.dataset)) * 100.0
            cw_dyn_acc = (cw_dyn_corr / len(test_loader.dataset)) * 100.0
            
            grand_metrics[d_name]['GIN'].append(gin_acc)
            grand_metrics[d_name]['StaticCW'].append(cw_static_acc)
            grand_metrics[d_name]['DynamicCW'].append(cw_dyn_acc)
            
            print(f"GIN: {gin_acc:.1f}% | StaticCW: {cw_static_acc:.1f}% | DynCW: {cw_dyn_acc:.1f}%")

    print("\n" + "="*80)
    print("GRAND CRUCIBLE COMPLETE. CALCULATING STATISTICS.")
    print("="*80)
    
    final_grand_json = {}
    for d_name, metrics in grand_metrics.items():
        final_grand_json[d_name] = {
            'GINBaseline': {'Mean_Accuracy': float(np.mean(metrics['GIN'])), 'Std_Dev': float(np.std(metrics['GIN']))},
            'StaticCWNetwork': {'Mean_Accuracy': float(np.mean(metrics['StaticCW'])), 'Std_Dev': float(np.std(metrics['StaticCW']))},
            'DynamicCWNetwork': {'Mean_Accuracy': float(np.mean(metrics['DynamicCW'])), 'Std_Dev': float(np.std(metrics['DynamicCW']))}
        }
        print(f"\n{d_name}:")
        print(f"  GIN: {final_grand_json[d_name]['GINBaseline']['Mean_Accuracy']:.2f}%")
        print(f"  StaticCW: {final_grand_json[d_name]['StaticCWNetwork']['Mean_Accuracy']:.2f}%")
        print(f"  DynamicCW: {final_grand_json[d_name]['DynamicCWNetwork']['Mean_Accuracy']:.2f}%")
        
    with open('grand_statistics.json', 'w') as f:
        json.dump(final_grand_json, f, indent=4)
        
    print("\nExported grand_statistics.json!")

if __name__ == "__main__":
    run_grand_crucible()
