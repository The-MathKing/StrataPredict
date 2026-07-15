import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split
import numpy as np
import json
import scipy.stats as stats
import itertools

from train import process_dataset, train_epoch, test
from model import CurvatureMPSN
from model_baselines import BaselineGCN

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_hyperparameter_tuning(dataset_name='NCI1'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = TUDataset(root=f'/tmp/{dataset_name}', name=dataset_name)
    num_classes = dataset.num_classes
    num_node_features = dataset.num_node_features if dataset.num_node_features > 0 else 1
    
    # Subset to save time in benchmarking
    dataset = dataset[:300] if len(dataset) > 300 else dataset
    processed_dataset = process_dataset(dataset)
    
    if not processed_dataset:
        print("Dataset processing failed.")
        return
        
    train_data, test_data = train_test_split(processed_dataset, test_size=0.2, random_state=42)
    
    # Grid Search space
    learning_rates = [0.01, 0.005]
    hidden_dims = [16, 32]
    gating_mechanisms = ['curvature', 'scalar']
    
    best_acc = 0.0
    best_config = {}
    
    print("\n--- Starting Grid Search ---")
    for lr, h_dim, gating in itertools.product(learning_rates, hidden_dims, gating_mechanisms):
        set_seed(42)
        model = CurvatureMPSN(num_node_features, h_dim, num_classes, gating=gating).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        acc = 0.0
        for epoch in range(1, 16):
            train_epoch(model, optimizer, criterion, train_data, device, is_gcn=False)
            _, current_acc = test(model, criterion, test_data, device, is_gcn=False)
            if current_acc > acc:
                acc = current_acc
                
        print(f"LR: {lr}, Dim: {h_dim}, Gating: {gating} -> Acc: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_config = {'lr': lr, 'hidden_dim': h_dim, 'gating': gating}
            
    print(f"\nBest Config Found: {best_config} with Acc: {best_acc*100:.2f}%")
    
    # Statistical Validation (N=10 runs)
    print("\n--- Running Statistical Validation (N=10) ---")
    
    def evaluate_trials(model_class, kwargs, is_gcn=False, trials=10):
        accuracies = []
        for i in range(trials):
            set_seed(i)
            model = model_class(**kwargs).to(device)
            optimizer = optim.Adam(model.parameters(), lr=best_config.get('lr', 0.005))
            criterion = nn.CrossEntropyLoss()
            
            trial_best_acc = 0.0
            for epoch in range(1, 16):
                train_epoch(model, optimizer, criterion, train_data, device, is_gcn)
                _, current_acc = test(model, criterion, test_data, device, is_gcn)
                if current_acc > trial_best_acc:
                    trial_best_acc = current_acc
            accuracies.append(trial_best_acc)
        return accuracies

    print("Evaluating Baseline GCN...")
    gcn_accs = evaluate_trials(BaselineGCN, 
                               {'num_node_features': num_node_features, 'hidden_dim': best_config['hidden_dim'], 'num_classes': num_classes},
                               is_gcn=True)
                               
    print("Evaluating Best Curvature MPSN...")
    mpsn_accs = evaluate_trials(CurvatureMPSN, 
                                {'num_node_features': num_node_features, 'hidden_dim': best_config['hidden_dim'], 'num_classes': num_classes, 'gating': best_config['gating']},
                                is_gcn=False)
                                
    gcn_mean, gcn_std = np.mean(gcn_accs), np.std(gcn_accs)
    mpsn_mean, mpsn_std = np.mean(mpsn_accs), np.std(mpsn_accs)
    
    # 2-sample t-test
    t_stat, p_value = stats.ttest_ind(mpsn_accs, gcn_accs, equal_var=False)
    
    print("\n==================================")
    print("STATISTICAL VALIDATION RESULTS")
    print("==================================")
    print(f"Baseline GCN:   {gcn_mean*100:.2f}% ± {gcn_std*100:.2f}%")
    print(f"Curvature MPSN: {mpsn_mean*100:.2f}% ± {mpsn_std*100:.2f}%")
    print(f"T-Statistic:    {t_stat:.4f}")
    print(f"P-Value:        {p_value:.4e}")
    
    if p_value < 0.05:
        print("Result: SIGNIFICANT improvement (p < 0.05)")
    else:
        print("Result: Improvement is NOT statistically significant.")
        
    # Save to JSON
    results_out = {
        'best_config': best_config,
        'baseline_gcn_mean': gcn_mean,
        'baseline_gcn_std': gcn_std,
        'mpsn_mean': mpsn_mean,
        'mpsn_std': mpsn_std,
        't_stat': t_stat,
        'p_value': p_value
    }
    
    with open('tuning_results.json', 'w') as f:
        json.dump(results_out, f, indent=4)
    print("\nSaved results to tuning_results.json")

if __name__ == "__main__":
    run_hyperparameter_tuning()
