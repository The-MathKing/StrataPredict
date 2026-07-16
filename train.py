"""
Training utilities module used during testing.
Contains helper functions to extract incidence matrices (B1, B2) from 
simplicial complexes, preprocess datasets, and execute standard training 
and testing loops for Graph Neural Networks.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split
import numpy as np
import time
import multiprocessing

from data_processing import lift_graph_to_simplicial_complex
from model import CurvatureMPSN
from model_baselines import BaselineGCN

def get_incidence_matrices(sc):
    """
    Extracts the B1 (nodes to edges) and B2 (edges to triangles) 
    incidence matrices from a TopoNetX SimplicialComplex as sparse PyTorch tensors.
    """
    # toponetx incidence matrices are scipy sparse matrices
    if sc.dim >= 1:
        B1_scipy = sc.incidence_matrix(rank=1, signed=True)
        # Convert scipy sparse to pytorch sparse
        coo = B1_scipy.tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        shape = coo.shape
        B1 = torch.sparse_coo_tensor(indices, values, size=shape).coalesce()
    else:
        B1 = torch.zeros((len(sc.skeleton(0)), 0))
        
    if sc.dim >= 2:
        B2_scipy = sc.incidence_matrix(rank=2, signed=True)
        coo = B2_scipy.tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        shape = coo.shape
        B2 = torch.sparse_coo_tensor(indices, values, size=shape).coalesce()
    else:
        B2 = torch.zeros((len(sc.skeleton(1)), 0))
        
    return B1, B2

def _process_single_graph(args):
    data, ignore_node_features = args
    sc, _ = lift_graph_to_simplicial_complex(data)
    
    B1, B2 = get_incidence_matrices(sc)
    
    if hasattr(data, 'x') and data.x is not None and not ignore_node_features:
        x_0 = data.x
    else:
        x_0 = torch.ones((len(sc.skeleton(0)), 1))
        
    if sc.dim >= 1:
        frc_dict = sc.get_simplex_attributes('frc')
        frc_list = [frc_dict[tuple(edge)] for edge in sc.skeleton(1)]
        frc_weights = torch.tensor(frc_list, dtype=torch.float32).unsqueeze(1)
    else:
        frc_weights = torch.empty((0, 1))
        
    return {
        'x_0': x_0,
        'edge_index': data.edge_index,
        'B1': B1,
        'B2': B2,
        'frc': frc_weights,
        'y': data.y
    }

def process_dataset(dataset, ignore_node_features=False):
    """
    Preprocess all graphs in the dataset into their topological representations.
    """
    processed_data = []
    print("Lifting graphs to Simplicial Complexes (Using 7 Cores)...")
    
    args_list = [(data, ignore_node_features) for data in dataset]
    
    pool = multiprocessing.Pool(processes=7)
    
    for i, result in enumerate(pool.imap(_process_single_graph, args_list)):
        processed_data.append(result)
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(dataset)} graphs")
            
    pool.close()
    pool.join()
            
    return processed_data

def train_epoch(model, optimizer, criterion, train_data, device, is_gcn=False):
    model.train()
    total_loss = 0
    correct = 0
    
    for data in train_data:
        optimizer.zero_grad()
        
        if is_gcn:
            # GCN needs x and edge_index
            out = model(data['x_0'].to(device), data['edge_index'].to(device))
        else:
            # MPSN needs incidence matrices and FRC
            out = model(data['x_0'].to(device), None, None, 
                        data['B1'].to(device), data['B2'].to(device), 
                        data['frc'].to(device), None, None, None)
            
        y = data['y'].to(device)
        
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum())
        
    return total_loss / len(train_data), correct / len(train_data)

def test(model, criterion, test_data, device, is_gcn=False):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data in test_data:
            if is_gcn:
                out = model(data['x_0'].to(device), data['edge_index'].to(device))
            else:
                out = model(data['x_0'].to(device), None, None, 
                            data['B1'].to(device), data['B2'].to(device), 
                            data['frc'].to(device), None, None, None)
                
            y = data['y'].to(device)
            loss = criterion(out, y)
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += int((pred == y).sum())
            
    return total_loss / len(test_data), correct / len(test_data)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading NCI1 dataset...")
    dataset = TUDataset(root='/tmp/NCI1', name='NCI1')
    
    print(f"Dataset Size: {len(dataset)} graphs")
    
    processed_dataset = process_dataset(dataset)
    num_node_features = dataset.num_node_features
    if num_node_features == 0:
        num_node_features = 1
        
    num_classes = dataset.num_classes
    
    # For a fast comparison, we do a single 80/20 train/test split
    train_data, test_data = train_test_split(processed_dataset, test_size=0.2, random_state=42)
    
    print(f"Training on {len(train_data)} graphs, Testing on {len(test_data)} graphs.")
    
    def train_and_eval(model, is_gcn, name):
        print(f"\n--- Training {name} ---")
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        epoch_runtimes = []
        best_test_acc = 0.0
        
        # 30 epochs for a fast benchmark
        for epoch in range(1, 31):
            start_time = time.time()
            train_loss, train_acc = train_epoch(model, optimizer, criterion, train_data, device, is_gcn)
            end_time = time.time()
            
            epoch_time = end_time - start_time
            epoch_runtimes.append(epoch_time)
            
            test_loss, test_acc = test(model, criterion, test_data, device, is_gcn)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Time: {epoch_time:.3f}s | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
                
        print(f"-> {name} Best Test Accuracy: {best_test_acc*100:.2f}%")
        print(f"-> {name} Avg Time/Epoch: {np.mean(epoch_runtimes):.3f}s")
        return best_test_acc
        
    # Initialize and train GCN Baseline
    gcn_model = BaselineGCN(num_node_features=num_node_features, hidden_dim=32, num_classes=num_classes).to(device)
    gcn_acc = train_and_eval(gcn_model, is_gcn=True, name="Standard GCN (1-WL Baseline)")
    
    # Initialize and train CurvatureMPSN
    mpsn_model = CurvatureMPSN(num_node_features=num_node_features, hidden_dim=32, num_classes=num_classes).to(device)
    mpsn_acc = train_and_eval(mpsn_model, is_gcn=False, name="Curvature-Weighted MPSN")
    
    print("\n==================================")
    print("FINAL COMPARISON (NCI1 DATASET)")
    print("==================================")
    print(f"Standard GCN:         {gcn_acc*100:.2f}%")
    print(f"Curvature MPSN:       {mpsn_acc*100:.2f}%")
    print(f"Improvement:          +{(mpsn_acc - gcn_acc)*100:.2f}%")
