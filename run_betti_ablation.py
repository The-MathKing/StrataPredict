import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from torch_geometric.data import Data
from data_processing import lift_graph_to_simplicial_complex
from train import get_incidence_matrices
from model import CurvatureMPSN
from model_baselines import BaselineGCN

device = torch.device('cpu')

def create_data(edges):
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    x = torch.ones((4, 1))
    y = torch.tensor([0])
    return Data(x=x, edge_index=edge_index, y=y)

edges_base = [(0,1), (1,2), (2,0), (0,3)]
data_base = create_data(edges_base)

edges_control = [(0,1), (1,2), (2,0)]
data_control = create_data(edges_control)

edges_topo = [(0,1), (2,0), (0,3)]
data_topo = create_data(edges_topo)

def process(data):
    sc, _ = lift_graph_to_simplicial_complex(data)
    B1, B2 = get_incidence_matrices(sc)
    
    if sc.dim >= 1:
        frc_dict = sc.get_simplex_attributes('frc')
        frc_list = [frc_dict[tuple(edge)] for edge in sc.skeleton(1)]
        frc_weights = torch.tensor(frc_list, dtype=torch.float32).unsqueeze(1)
    else:
        frc_weights = torch.empty((0, 1))
        
    return {
        'x_0': data.x,
        'edge_index': data.edge_index,
        'B1': B1,
        'B2': B2,
        'frc': frc_weights,
        'batch_0': torch.zeros(data.x.shape[0], dtype=torch.long),
        'batch_1': torch.zeros(max(1, B1.shape[1]), dtype=torch.long),
        'batch_2': torch.zeros(max(1, B2.shape[1]), dtype=torch.long)
    }

base_input = process(data_base)
control_input = process(data_control)
topo_input = process(data_topo)

torch.manual_seed(42)
mpsn = CurvatureMPSN(num_node_features=1, hidden_dim=32, num_classes=2, gating='vector').to(device)
gcn = BaselineGCN(num_node_features=1, hidden_dim=32, num_classes=2).to(device)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
mpsn.apply(init_weights)
gcn.apply(init_weights)

mpsn.eval()
gcn.eval()

with torch.no_grad():
    gcn_base = gcn(base_input['x_0'], base_input['edge_index'])
    gcn_control = gcn(control_input['x_0'], control_input['edge_index'])
    gcn_topo = gcn(topo_input['x_0'], topo_input['edge_index'])
    
    mpsn_base = mpsn(base_input['x_0'], None, None, base_input['B1'], base_input['B2'], base_input['frc'], base_input['batch_0'], base_input['batch_1'], base_input['batch_2'])
    mpsn_control = mpsn(control_input['x_0'], None, None, control_input['B1'], control_input['B2'], control_input['frc'], control_input['batch_0'], control_input['batch_1'], control_input['batch_2'])
    mpsn_topo = mpsn(topo_input['x_0'], None, None, topo_input['B1'], topo_input['B2'], topo_input['frc'], topo_input['batch_0'], topo_input['batch_1'], topo_input['batch_2'])

gcn_dist_control = torch.norm(gcn_base - gcn_control).item()
gcn_dist_topo = torch.norm(gcn_base - gcn_topo).item()
gcn_ratio = gcn_dist_topo / (gcn_dist_control + 1e-9)

mpsn_dist_control = torch.norm(mpsn_base - mpsn_control).item()
mpsn_dist_topo = torch.norm(mpsn_base - mpsn_topo).item()
mpsn_ratio = mpsn_dist_topo / (mpsn_dist_control + 1e-9)

print(f"GCN Control Delta: {gcn_dist_control:.4f}")
print(f"GCN Topo Delta:    {gcn_dist_topo:.4f}")
print(f"GCN Sensitivity Ratio (Topo/Control): {gcn_ratio:.2f}x\n")

print(f"MPSN Control Delta: {mpsn_dist_control:.4f}")
print(f"MPSN Topo Delta:    {mpsn_dist_topo:.4f}")
print(f"MPSN Sensitivity Ratio (Topo/Control): {mpsn_ratio:.2f}x")
