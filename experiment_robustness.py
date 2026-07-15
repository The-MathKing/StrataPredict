import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split
import numpy as np

from adversarial_utils import inject_edge_noise, perturb_cycles
from train import process_dataset, train_epoch, test
from model import CurvatureMPSN
from model_baselines import BaselineGCN

def perturb_dataset(dataset, p=0.0, mode='drop', cycle_length=0):
    perturbed_data_list = []
    print(f"Applying perturbation: mode={mode}, p={p}, cycle_length={cycle_length}")
    for i, data in enumerate(dataset):
        try:
            if cycle_length > 0:
                new_data = perturb_cycles(data, cycle_length)
            else:
                new_data = inject_edge_noise(data, p=p, mode=mode)
            perturbed_data_list.append(new_data)
        except Exception as e:
            pass
    return perturbed_data_list

def evaluate_robustness(dataset_name='NCI1', noise_levels=[0.0, 0.05, 0.1, 0.2]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = TUDataset(root=f'/tmp/{dataset_name}', name=dataset_name)
    num_classes = dataset.num_classes
    num_node_features = dataset.num_node_features
    if num_node_features == 0:
        num_node_features = 1
        
    dataset = dataset[:200] if len(dataset) > 200 else dataset
        
    results = {'GCN': [], 'MPSN': []}
    
    for p in noise_levels:
        print(f"\n--- Noise Level p={p} ---")
        perturbed_dataset = perturb_dataset(dataset, p=p, mode='drop')
        
        # Process the perturbed dataset to build simplicial complexes
        processed_dataset = process_dataset(perturbed_dataset)
        if not processed_dataset:
            print("Failed to process dataset. Skipping.")
            continue
            
        train_data, test_data = train_test_split(processed_dataset, test_size=0.2, random_state=42)
        
        def train_eval_model(model, is_gcn):
            optimizer = optim.Adam(model.parameters(), lr=0.005)
            criterion = nn.CrossEntropyLoss()
            best_acc = 0.0
            for epoch in range(1, 16): # 15 epochs for rapid test
                train_epoch(model, optimizer, criterion, train_data, device, is_gcn)
                _, test_acc = test(model, criterion, test_data, device, is_gcn)
                if test_acc > best_acc:
                    best_acc = test_acc
            return best_acc
            
        gcn_model = BaselineGCN(num_node_features=num_node_features, hidden_dim=32, num_classes=num_classes).to(device)
        gcn_acc = train_eval_model(gcn_model, is_gcn=True)
        results['GCN'].append((p, gcn_acc))
        print(f"GCN Test Acc: {gcn_acc*100:.2f}%")
        
        mpsn_model = CurvatureMPSN(num_node_features=num_node_features, hidden_dim=32, num_classes=num_classes).to(device)
        mpsn_acc = train_eval_model(mpsn_model, is_gcn=False)
        results['MPSN'].append((p, mpsn_acc))
        print(f"MPSN Test Acc: {mpsn_acc*100:.2f}%")

    print("\n==================================")
    print("ROBUSTNESS EVALUATION RESULTS")
    print("==================================")
    for i, p in enumerate(noise_levels):
        gcn_acc = results['GCN'][i][1]
        mpsn_acc = results['MPSN'][i][1]
        print(f"Noise p={p:.2f} -> GCN: {gcn_acc*100:.2f}%, MPSN: {mpsn_acc*100:.2f}% (Improvement: +{(mpsn_acc - gcn_acc)*100:.2f}%)")

if __name__ == "__main__":
    evaluate_robustness()
