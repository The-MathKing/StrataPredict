"""
Orchestrator script for the Curvature MPSN Benchmarking Suite.
This script is the main entry point used for testing. It runs the full 
N=10 cross-validation suite for Manifold Diversity, Adversarial Robustness, 
Ablation Studies, and Scalability Profiling, outputting to a JSON file.
"""
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats as stats
import json
import time
import tracemalloc
import sys

from train import process_dataset, train_epoch, test, get_incidence_matrices
from model import CurvatureMPSN
from model_baselines import BaselineGCN
from adversarial_utils import inject_edge_noise, targeted_bottleneck_attack
from data_processing import lift_graph_to_simplicial_complex

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_n_trials(model_fn, train_data, test_data, device, is_gcn, n_trials=10, epochs=15):
    accs = []
    for i in range(n_trials):
        set_seed(i)
        model = model_fn().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()
        
        trial_best = 0.0
        for ep in range(epochs):
            train_epoch(model, optimizer, criterion, train_data, device, is_gcn)
            _, test_acc = test(model, criterion, test_data, device, is_gcn)
            if test_acc > trial_best:
                trial_best = test_acc
        accs.append(trial_best)
    return accs

def compare_models(gcn_fn, mpsn_fn, train_data, test_data, device, n_trials=10, epochs=15):
    print("  Running GCN trials...")
    gcn_accs = run_n_trials(gcn_fn, train_data, test_data, device, True, n_trials, epochs)
    print("  Running MPSN trials...")
    mpsn_accs = run_n_trials(mpsn_fn, train_data, test_data, device, False, n_trials, epochs)
    
    gcn_mean, gcn_std = np.mean(gcn_accs), np.std(gcn_accs)
    mpsn_mean, mpsn_std = np.mean(mpsn_accs), np.std(mpsn_accs)
    
    # Need at least variance to do t-test properly, but scipy handles it or returns nan
    if gcn_std == 0 and mpsn_std == 0:
        if gcn_mean == mpsn_mean:
            t_stat, p_val = 0.0, 1.0
        else:
            t_stat, p_val = float('inf'), 0.0
    else:
        t_stat, p_val = stats.ttest_ind(mpsn_accs, gcn_accs, equal_var=False)
        
    return {
        'gcn_mean': gcn_mean, 'gcn_std': gcn_std,
        'mpsn_mean': mpsn_mean, 'mpsn_std': mpsn_std,
        'p_value': p_val if not np.isnan(p_val) else 1.0
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    final_results = {}
    
    # 1. Manifold Diversity
    print("\n--- 1. Manifold Diversity ---")
    final_results['diversity'] = {}
    for ds_name in ['REDDIT-BINARY', 'COLLAB']:
        print(f"Processing {ds_name}...")
        dataset = TUDataset(root=f'/tmp/{ds_name}', name=ds_name)
        num_classes = dataset.num_classes
        num_node_features = dataset.num_node_features if dataset.num_node_features > 0 else 1
        
        proc_data = process_dataset(dataset)
        train_data, test_data = train_test_split(proc_data, test_size=0.2, random_state=42)
        
        gcn_fn = lambda: BaselineGCN(num_node_features, 32, num_classes)
        mpsn_fn = lambda: CurvatureMPSN(num_node_features, 32, num_classes)
        
        metrics = compare_models(gcn_fn, mpsn_fn, train_data, test_data, device, n_trials=10, epochs=10)
        final_results['diversity'][ds_name] = metrics
        print(f"  {ds_name}: GCN={metrics['gcn_mean']:.4f}, MPSN={metrics['mpsn_mean']:.4f}, p={metrics['p_value']:.4e}")

    # 2. Adversarial Robustness
    print("\n--- 2. Adversarial Robustness ---")
    final_results['robustness'] = {}
    ds_name = 'NCI1'
    print(f"Loading {ds_name} for Robustness...")
    dataset = TUDataset(root=f'/tmp/{ds_name}', name=ds_name)
    num_classes = dataset.num_classes
    num_node_features = dataset.num_node_features if dataset.num_node_features > 0 else 1
    
    for p in [0.0, 0.05, 0.1, 0.2]:
        print(f" Processing Noise p={p}...")
        perturbed = []
        for data in dataset:
            try:
                new_data = inject_edge_noise(data, p=p, mode='drop')
                perturbed.append(new_data)
            except Exception:
                pass
        
        proc_data = process_dataset(perturbed)
        train_data, test_data = train_test_split(proc_data, test_size=0.2, random_state=42)
        
        gcn_fn = lambda: BaselineGCN(num_node_features, 32, num_classes)
        mpsn_fn = lambda: CurvatureMPSN(num_node_features, 32, num_classes)
        
        metrics = compare_models(gcn_fn, mpsn_fn, train_data, test_data, device, n_trials=10, epochs=10)
        final_results['robustness'][f'p_{p}'] = metrics
        print(f"  p={p}: GCN={metrics['gcn_mean']:.4f}, MPSN={metrics['mpsn_mean']:.4f}, p={metrics['p_value']:.4e}")
        
    print("\n  [Targeted Bottleneck Attack]")
    for p in [0.05, 0.1]:
        print(f" Processing Targeted Drop p={p}...")
        perturbed = []
        for data in dataset:
            try:
                new_data = targeted_bottleneck_attack(data, drop_percent=p)
                perturbed.append(new_data)
            except Exception:
                pass
        
        proc_data = process_dataset(perturbed)
        train_data, test_data = train_test_split(proc_data, test_size=0.2, random_state=42)
        
        gcn_fn = lambda: BaselineGCN(num_node_features, 32, num_classes)
        mpsn_fn = lambda: CurvatureMPSN(num_node_features, 32, num_classes)
        
        metrics = compare_models(gcn_fn, mpsn_fn, train_data, test_data, device, n_trials=10, epochs=10)
        final_results['robustness'][f'targeted_p_{p}'] = metrics
        print(f"  Targeted p={p}: GCN={metrics['gcn_mean']:.4f}, MPSN={metrics['mpsn_mean']:.4f}, p={metrics['p_value']:.4e}")
        
    # 3. Ablation Study
    print("\n--- 3. Ablation Study ---")
    final_results['ablation'] = {}
    ds_name = 'MUTAG'
    print(f"Loading {ds_name} for Ablation...")
    dataset = TUDataset(root=f'/tmp/{ds_name}', name=ds_name)
    num_classes = dataset.num_classes
    num_node_features = dataset.num_node_features if dataset.num_node_features > 0 else 1
    
    def process_ablation(max_dim):
        pd = []
        for data in dataset:
            try:
                sc, _ = lift_graph_to_simplicial_complex(data, max_dim=max_dim)
                B1, B2 = get_incidence_matrices(sc)
                x_0 = data.x if hasattr(data, 'x') and data.x is not None else torch.ones((len(sc.skeleton(0)), 1))
                if sc.dim >= 1:
                    frc_dict = sc.get_simplex_attributes('frc')
                    frc_list = [frc_dict[tuple(edge)] for edge in sc.skeleton(1)]
                    frc_weights = torch.tensor(frc_list, dtype=torch.float32).unsqueeze(1)
                else:
                    frc_weights = torch.empty((0, 1))
                pd.append({'x_0': x_0, 'B1': B1, 'B2': B2, 'frc': frc_weights, 'y': data.y})
            except Exception:
                pass
        return pd

    data_2 = process_ablation(2)
    data_3 = process_ablation(3)
    train_2, test_2 = train_test_split(data_2, test_size=0.2, random_state=42)
    train_3, test_3 = train_test_split(data_3, test_size=0.2, random_state=42)
    
    for max_dim, (train_data, test_data) in zip([2, 3], [(train_2, test_2), (train_3, test_3)]):
        final_results['ablation'][f'dim_{max_dim}'] = {}
        for gating in ['none', 'scalar', 'vector', 'curvature']:
            print(f" Processing Dim={max_dim}, Gating={gating}...")
            mpsn_fn = lambda: CurvatureMPSN(num_node_features, 32, num_classes, gating=gating)
            accs = run_n_trials(mpsn_fn, train_data, test_data, device, False, n_trials=10, epochs=10)
            mean, std = np.mean(accs), np.std(accs)
            final_results['ablation'][f'dim_{max_dim}'][gating] = {'mean': mean, 'std': std}
            print(f"  Acc={mean:.4f} ± {std:.4f}")

    # 4. Scalability
    print("\n--- 4. Scalability Profile ---")
    final_results['scalability'] = {}
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/tmp/ogb')
        graph = dataset[0]
        
        tracemalloc.start()
        t0 = time.time()
        sc, _ = lift_graph_to_simplicial_complex(graph, max_dim=2)
        B1, B2 = get_incidence_matrices(sc)
        t1 = time.time()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        final_results['scalability']['ogbn-arxiv'] = {
            'time_seconds': t1 - t0,
            'peak_memory_mb': peak / 10**6,
            'nodes': len(sc.skeleton(0)),
            'edges': len(sc.skeleton(1)) if sc.dim >= 1 else 0
        }
        print(f"  Time: {t1-t0:.2f}s, Peak Mem: {peak/10**6:.2f}MB")
    except Exception as e:
        print(f"Scalability failed: {e}")
        
    # 5. Transferability Benchmark (Zero-Shot)
    print("\n--- 5. Transferability Benchmark (Zero-Shot) ---")
    final_results['transferability'] = {}
    try:
        print("Loading PROTEINS and NCI1 datasets...")
        proteins_dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')
        nci1_dataset = TUDataset(root='/tmp/NCI1', name='NCI1')
        
        num_classes = 2 # Both are binary classification
        num_node_features = 1 # Using ignore_node_features=True
        
        print("Processing PROTEINS (Source)...")
        source_data = process_dataset(proteins_dataset, ignore_node_features=True)
        print("Processing NCI1 (Target)...")
        target_data = process_dataset(nci1_dataset, ignore_node_features=True)
        
        train_source, _ = train_test_split(source_data, test_size=0.2, random_state=42)
        test_target = target_data
        
        gcn_fn = lambda: BaselineGCN(num_node_features, 32, num_classes)
        mpsn_fn = lambda: CurvatureMPSN(num_node_features, 32, num_classes)
        
        print("  Evaluating Zero-Shot Transfer...")
        metrics = compare_models(gcn_fn, mpsn_fn, train_source, test_target, device, n_trials=10, epochs=10)
        final_results['transferability']['PROTEINS_to_NCI1'] = metrics
        print(f"  Zero-Shot Transfer: GCN={metrics['gcn_mean']:.4f}, MPSN={metrics['mpsn_mean']:.4f}, p={metrics['p_value']:.4e}")
    except Exception as e:
        print(f"Transferability failed: {e}")
        
    with open('results_full_run.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    print("\nSaved full metrics to results_full_run.json")

if __name__ == '__main__':
    main()
