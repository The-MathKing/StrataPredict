import warnings
warnings.filterwarnings("ignore")
import torch
import networkx as nx
from torch_geometric.utils import from_networkx, to_networkx
import time
import numpy as np
import json
from torch_geometric.datasets import TUDataset
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split

from data_processing import lift_graph_to_simplicial_complex
from train import get_incidence_matrices, process_dataset, train_epoch, test
from model import CurvatureMPSN
from run_all_benchmarks_v2 import generate_srgs, process_graph

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def test_isomorphism():
    G1, G2 = generate_srgs()
    x0_1, B1_1, B2_1, frc_1 = process_graph(G1)
    x0_2, B1_2, B2_2, frc_2 = process_graph(G2)
    
    set_seed(42)
    model = CurvatureMPSN(num_node_features=1, hidden_dim=32, num_classes=2, gating='curvature')
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=1.0)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=1.0)
    model.apply(init_weights)
    model.eval()
    
    with torch.no_grad():
        out1 = model(x0_1, None, None, B1_1, B2_1, frc_1, None, None, None)
        out2 = model(x0_2, None, None, B1_2, B2_2, frc_2, None, None, None)
        
    distance = torch.norm(out1 - out2, p=2).item()
    return distance

def test_training():
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    # Use only 20 graphs for light test
    subset = []
    for d in dataset:
        subset.append(d)
        if len(subset) >= 20: break
        
    proc_data = process_dataset(subset)
    train_data, test_data = train_test_split(proc_data, test_size=0.5, random_state=42)
    
    set_seed(42)
    model = CurvatureMPSN(dataset.num_node_features if dataset.num_node_features > 0 else 1, 32, dataset.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    train_epoch(model, optimizer, criterion, train_data, torch.device('cpu'), False)
    loss, acc = test(model, criterion, test_data, torch.device('cpu'), False)
    
    return acc

def test_latency():
    dataset = TUDataset(root='/tmp/NCI1', name='NCI1')
    data = dataset[0]
            
    x0, B1, B2, frc = process_graph(to_networkx(data, to_undirected=True))
    model = CurvatureMPSN(num_node_features=1, hidden_dim=32, num_classes=2, gating='curvature')
    model.eval()
    
    with torch.no_grad():
        x_0 = model.node_embedding(x0)
        x_1 = model.edge_embedding(torch.ones((B1.shape[1], 1)))
        x_2 = model.triangle_embedding(torch.ones((B2.shape[1], 1)))
    
    for _ in range(5):
        _ = model.conv1(x_0, x_1, x_2, B1, B2, frc)
        
    t0 = time.time()
    with torch.no_grad():
        o0, o1, o2 = model.conv1(x_0, x_1, x_2, B1, B2, frc)
    t1 = time.time()
    
    t2 = time.time()
    with torch.no_grad():
        model.conv2(o0, o1, o2, B1, B2, frc)
    t3 = time.time()
    
    return {"layer_1_ms": (t1 - t0)*1000, "layer_2_ms": (t3 - t2)*1000}

def main():
    results = {}
    print("Running light Isomorphism Test...")
    results['isomorphism_l2_distance'] = test_isomorphism()
    print("Running light 1-Epoch Training Test on MUTAG...")
    results['quick_training_acc'] = test_training()
    print("Running light Latency Test...")
    results['latency'] = test_latency()
    
    with open('light_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nLight tests complete! Results saved to light_results.json")

if __name__ == '__main__':
    main()
