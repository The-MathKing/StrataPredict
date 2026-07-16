import os
import torch
import random
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from data_processing import compute_forman_ricci_curvature
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def visualize_molecules(num_samples=5):
    print("Loading NCI1 dataset for visualization...")
    dataset = TUDataset(root='/tmp/NCI1', name='NCI1')
    
    os.makedirs('figures', exist_ok=True)
    
    indices = random.sample(range(len(dataset)), num_samples)
    
    for i, idx in enumerate(indices):
        data = dataset[idx]
        G = to_networkx(data, to_undirected=True)
        
        edge_colors = []
        frcs = []
        edges = list(G.edges())
        for edge in edges:
            frc = compute_forman_ricci_curvature(G, edge)
            frcs.append(frc)
            
        if not frcs:
            print(f"Graph {idx} has no edges, skipping.")
            continue
            
        max_abs_frc = max(max(frcs), abs(min(frcs))) if frcs else 1.0
        if max_abs_frc == 0:
            max_abs_frc = 1.0
            
        # Red is usually for negative, Blue for positive in RdBu, but we want 
        # to ensure it behaves correctly. RdBu means Red to Blue.
        # Actually RdBu_r is better if we want Red for negative (low values).
        norm = mcolors.TwoSlopeNorm(vmin=-max_abs_frc, vcenter=0., vmax=max_abs_frc)
        cmap = plt.get_cmap('RdBu') 
        
        for frc in frcs:
            edge_colors.append(cmap(norm(frc)))
            
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightgray')
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=3.0)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label="Forman-Ricci Curvature")
        
        plt.title(f"NCI1 Molecule {idx} - FRC Heatmap")
        plt.axis('off')
        
        filepath = f"figures/nci1_molecule_{idx}_frc.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {filepath}")

if __name__ == "__main__":
    visualize_molecules()
