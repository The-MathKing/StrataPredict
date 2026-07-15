import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split
import numpy as np

from data_processing import lift_graph_to_simplicial_complex
from train import get_incidence_matrices
from model import CurvatureMPSN

def process_dataset_ablated(dataset, max_dim=2):
    processed_data = []
    print(f"Lifting graphs with max_dim={max_dim}...")
    for i, data in enumerate(dataset):
        try:
            sc, _ = lift_graph_to_simplicial_complex(data, max_dim=max_dim)
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
                
            processed_data.append({
                'x_0': x_0,
                'B1': B1,
                'B2': B2,
                'frc': frc_weights,
                'y': data.y
            })
        except Exception as e:
            pass
    return processed_data

def run_advanced_ablation(dataset_name='MUTAG'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = TUDataset(root=f'/tmp/{dataset_name}', name=dataset_name)
    num_classes = dataset.num_classes
    num_node_features = dataset.num_node_features if dataset.num_node_features > 0 else 1
    
    # Process for max_dim=2 and max_dim=3
    data_dim2 = process_dataset_ablated(dataset, max_dim=2)
    data_dim3 = process_dataset_ablated(dataset, max_dim=3)
    
    train_2, test_2 = train_test_split(data_dim2, test_size=0.2, random_state=42)
    train_3, test_3 = train_test_split(data_dim3, test_size=0.2, random_state=42)
    
    def train_eval(train_data, test_data, gating, max_dim):
        model = CurvatureMPSN(num_node_features, 32, num_classes, gating=gating).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
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
                
            model.eval()
            correct = 0
            with torch.no_grad():
                for data in test_data:
                    out = model(data['x_0'].to(device), None, None, 
                                data['B1'].to(device), data['B2'].to(device), 
                                data['frc'].to(device), None, None, None)
                    pred = out.argmax(dim=1)
                    correct += int((pred == data['y'].to(device)).sum())
            test_acc = correct / len(test_data)
            if test_acc > best_acc:
                best_acc = test_acc
        return best_acc

    results = []
    
    for max_dim, (train_data, test_data) in zip([2, 3], [(train_2, test_2), (train_3, test_3)]):
        for gating in ['none', 'scalar', 'vector', 'curvature']:
            acc = train_eval(train_data, test_data, gating, max_dim)
            print(f"Dim: {max_dim}, Gating: {gating:10} -> Acc: {acc*100:.2f}%")
            results.append((max_dim, gating, acc))
            
    print("\n==================================")
    print("ADVANCED ABLATION RESULTS")
    print("==================================")
    for dim, gating, acc in results:
        print(f"Max-Dim {dim} | Gating: {gating:10} | Acc: {acc*100:.2f}%")

if __name__ == "__main__":
    run_advanced_ablation()
