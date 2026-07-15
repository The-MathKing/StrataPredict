import torch
import torch.nn as nn
import torch.optim as optim
import time
import tracemalloc
import sys

# Attempt to import ogb, if not installed we will skip gracefully
try:
    from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
    OGB_AVAILABLE = True
except ImportError:
    OGB_AVAILABLE = False
    print("WARNING: 'ogb' package not installed. Scalability script will just be a template.")

from data_processing import lift_graph_to_simplicial_complex
from train import get_incidence_matrices
from model import CurvatureMPSN

def profile_scalability(dataset_name='ogbn-arxiv'):
    if not OGB_AVAILABLE:
        print("Please install ogb: pip install ogb")
        return
        
    print(f"\n{'='*50}\nProfiling Scalability on {dataset_name}\n{'='*50}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load dataset
    print("Loading dataset...")
    t0 = time.time()
    dataset = PygNodePropPredDataset(name=dataset_name, root='/tmp/ogb')
    graph = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    print(f"Dataset loaded in {time.time()-t0:.2f}s")
    print(f"Graph nodes: {graph.num_nodes}, edges: {graph.num_edges}")
    
    # To prevent out-of-memory on massive graphs in this quick benchmark, we subgraph
    # but for true scalability we'd measure the full thing. Here we just measure preprocessing time.
    
    print("\nStarting memory tracking for Cellular Lifting...")
    tracemalloc.start()
    t_start = time.time()
    
    # For speed in this automated script, we limit max_dim to 2.
    sc, _ = lift_graph_to_simplicial_complex(graph, max_dim=2)
    B1, B2 = get_incidence_matrices(sc)
    
    t_end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\n--- Preprocessing Profiling ---")
    print(f"Time to lift to simplicial complex: {t_end - t_start:.2f} seconds")
    print(f"Peak Memory usage: {peak / 10**6:.2f} MB")
    
    print(f"Simplicial Complex shape:")
    print(f"  0-cells (nodes): {len(sc.skeleton(0))}")
    if sc.dim >= 1:
        print(f"  1-cells (edges): {len(sc.skeleton(1))}")
    if sc.dim >= 2:
        print(f"  2-cells (triangles): {len(sc.skeleton(2))}")
        
    # Set up model and run 1 epoch to profile forward pass
    num_node_features = graph.num_node_features if graph.num_node_features > 0 else 1
    num_classes = dataset.num_classes
    
    model = CurvatureMPSN(num_node_features, hidden_dim=32, num_classes=num_classes).to(device)
    
    if hasattr(graph, 'x') and graph.x is not None:
        x_0 = graph.x
    else:
        x_0 = torch.ones((len(sc.skeleton(0)), 1))
        
    if sc.dim >= 1:
        frc_dict = sc.get_simplex_attributes('frc')
        frc_list = [frc_dict[tuple(edge)] for edge in sc.skeleton(1)]
        frc_weights = torch.tensor(frc_list, dtype=torch.float32).unsqueeze(1)
    else:
        frc_weights = torch.empty((0, 1))
        
    y = graph.y.squeeze() if hasattr(graph, 'y') else torch.zeros(len(x_0), dtype=torch.long)
    
    x_0 = x_0.to(device)
    B1 = B1.to(device)
    B2 = B2.to(device)
    frc_weights = frc_weights.to(device)
    y = y.to(device)
    train_idx = train_idx.to(device)
    
    print("\n--- Forward Pass Profiling (1 Epoch) ---")
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    t_forward_start = time.time()
    
    optimizer.zero_grad()
    # No batched data, single giant graph
    out = model(x_0, None, None, B1, B2, frc_weights, None, None, None)
    
    loss = criterion(out[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()
    
    t_forward_end = time.time()
    
    print(f"Forward + Backward Pass Time: {t_forward_end - t_forward_start:.3f} seconds")
    if torch.cuda.is_available():
        print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        
    print("\nScalability profile complete. Data is ready for complexity plotting.")

if __name__ == "__main__":
    profile_scalability()
