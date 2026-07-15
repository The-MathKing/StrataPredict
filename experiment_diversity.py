import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset, PPI
from sklearn.model_selection import train_test_split
import numpy as np
import time
import argparse
import os

from data_processing import lift_graph_to_simplicial_complex
from model import CurvatureMPSN
from model_baselines import BaselineGCN
from train import get_incidence_matrices

def process_dataset(dataset):
    processed_data = []
    print(f"Lifting {len(dataset)} graphs to Simplicial Complexes...")
    for i, data in enumerate(dataset):
        try:
            sc, _ = lift_graph_to_simplicial_complex(data)
            B1, B2 = get_incidence_matrices(sc)
            
            if hasattr(data, 'x') and data.x is not None:
                x_0 = data.x
            else:
                x_0 = torch.ones((len(sc.skeleton(0)), 1))
                
            if sc.dim >= 1:
                frc_dict = sc.get_simplex_attributes('frc')
                frc_list = [frc_dict[tuple(edge)] for edge in sc.skeleton(1)]
                frc_weights = torch.tensor(frc_list, dtype=torch.float32).unsqueeze(1)
            else:
                frc_weights = torch.empty((0, 1))
                
            y = data.y if hasattr(data, 'y') else torch.zeros(1, dtype=torch.long)
            
            processed_data.append({
                'x_0': x_0,
                'edge_index': data.edge_index,
                'B1': B1,
                'B2': B2,
                'frc': frc_weights,
                'y': y
            })
            
            if (i+1) % 100 == 0:
                print(f"Processed {i+1}/{len(dataset)} graphs")
        except Exception as e:
            print(f"Skipping graph {i} due to error: {e}")
            continue
            
    return processed_data

def train_epoch(model, optimizer, criterion, train_data, device, is_gcn=False):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    for data in train_data:
        optimizer.zero_grad()
        
        if is_gcn:
            out = model(data['x_0'].to(device), data['edge_index'].to(device))
        else:
            out = model(data['x_0'].to(device), None, None, 
                        data['B1'].to(device), data['B2'].to(device), 
                        data['frc'].to(device), None, None, None)
            
        y = data['y'].to(device)
        if len(y.shape) > 1 and y.shape[1] > 1: # PPI or multi-label
            loss = nn.BCEWithLogitsLoss()(out, y.float())
            pred = (out > 0).float()
            correct += int((pred == y).sum())
            total_samples += y.numel()
        else:
            loss = criterion(out, y)
            pred = out.argmax(dim=1)
            correct += int((pred == y).sum())
            total_samples += y.size(0)
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_data), correct / max(1, total_samples)

def test(model, criterion, test_data, device, is_gcn=False):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data in test_data:
            if is_gcn:
                out = model(data['x_0'].to(device), data['edge_index'].to(device))
            else:
                out = model(data['x_0'].to(device), None, None, 
                            data['B1'].to(device), data['B2'].to(device), 
                            data['frc'].to(device), None, None, None)
                
            y = data['y'].to(device)
            if len(y.shape) > 1 and y.shape[1] > 1:
                loss = nn.BCEWithLogitsLoss()(out, y.float())
                pred = (out > 0).float()
                correct += int((pred == y).sum())
                total_samples += y.numel()
            else:
                loss = criterion(out, y)
                pred = out.argmax(dim=1)
                correct += int((pred == y).sum())
                total_samples += y.size(0)
                
            total_loss += loss.item()
            
    return total_loss / len(test_data), correct / max(1, total_samples)

def run_experiment(dataset_name, device):
    print(f"\n{'='*50}\nEvaluating on Dataset: {dataset_name}\n{'='*50}")
    
    if dataset_name == 'PPI':
        dataset = PPI(root='/tmp/PPI')
        num_classes = dataset.num_classes
        num_node_features = dataset.num_node_features
    else:
        dataset = TUDataset(root=f'/tmp/{dataset_name}', name=dataset_name)
        num_classes = dataset.num_classes
        num_node_features = dataset.num_node_features
        if num_node_features == 0:
            num_node_features = 1
            
    # For quick evaluation, limit dataset size if it's very large
    max_graphs = 500
    dataset_subset = dataset[:max_graphs] if len(dataset) > max_graphs else dataset
    print(f"Using {len(dataset_subset)} graphs (Subset size limited for benchmark)")
    
    processed_dataset = process_dataset(dataset_subset)
    if not processed_dataset:
        print("Failed to process any graphs.")
        return
        
    train_data, test_data = train_test_split(processed_dataset, test_size=0.2, random_state=42)
    
    def train_and_eval(model, is_gcn, name):
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0
        for epoch in range(1, 21):
            train_epoch(model, optimizer, criterion, train_data, device, is_gcn)
            _, test_acc = test(model, criterion, test_data, device, is_gcn)
            if test_acc > best_acc:
                best_acc = test_acc
        print(f"{name} Best Accuracy: {best_acc*100:.2f}%")
        return best_acc

    gcn_model = BaselineGCN(num_node_features=num_node_features, hidden_dim=32, num_classes=num_classes).to(device)
    gcn_acc = train_and_eval(gcn_model, is_gcn=True, name="Baseline GCN")
    
    mpsn_model = CurvatureMPSN(num_node_features=num_node_features, hidden_dim=32, num_classes=num_classes).to(device)
    mpsn_acc = train_and_eval(mpsn_model, is_gcn=False, name="Curvature MPSN")
    
    print(f"\nImprovement on {dataset_name}: +{(mpsn_acc - gcn_acc)*100:.2f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['REDDIT-BINARY', 'COLLAB', 'PPI'], help='Datasets to evaluate')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for ds in args.datasets:
        run_experiment(ds, device)
