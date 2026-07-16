import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split
from train import process_dataset, train_epoch, test
from model import CurvatureMPSN
from model_baselines import BaselineGCN
from adversarial_utils import targeted_bottleneck_attack

def dry_run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing Targeted Attack...")
    dataset = TUDataset(root='/tmp/NCI1', name='NCI1')
    
    small_ds = [dataset[i] for i in range(5)]
    
    perturbed = []
    for data in small_ds:
        new_data = targeted_bottleneck_attack(data, drop_percent=0.05)
        perturbed.append(new_data)
        
    proc_data = process_dataset(perturbed)
    
    print("Testing Transferability...")
    proteins = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')
    small_prot = [proteins[i] for i in range(5)]
    
    proc_prot = process_dataset(small_prot, ignore_node_features=True)
    proc_nci1 = process_dataset(small_ds, ignore_node_features=True)
    
    model = CurvatureMPSN(num_node_features=1, hidden_dim=32, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    train_epoch(model, optimizer, criterion, proc_prot, device, is_gcn=False)
    test(model, criterion, proc_nci1, device, is_gcn=False)
    print("All dry runs successful!")

if __name__ == '__main__':
    dry_run()
