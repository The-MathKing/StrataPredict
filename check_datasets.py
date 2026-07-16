import torch
from torch_geometric.datasets import TUDataset

def check_ds(name):
    ds = TUDataset(root=f'/tmp/{name}', name=name)
    print(f"--- {name} ---")
    print(f"Classes: {ds.num_classes}")
    print(f"Features: {ds.num_node_features}")
    
check_ds('PROTEINS')
check_ds('NCI1')
