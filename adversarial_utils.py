import torch
import random
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx

def inject_edge_noise(pyg_data, p=0.1, mode='drop'):
    """
    Injects structural noise into the graph.
    mode='drop': randomly drops p% of edges.
    mode='add': randomly adds edges equal to p% of existing edges.
    mode='flip': combination of drop and add.
    """
    G = to_networkx(pyg_data, to_undirected=True)
    num_edges_to_change = int(G.number_of_edges() * p)
    
    if mode in ['drop', 'flip']:
        edges = list(G.edges())
        if num_edges_to_change > 0:
            edges_to_drop = random.sample(edges, min(num_edges_to_change, len(edges)))
            G.remove_edges_from(edges_to_drop)
            
    if mode in ['add', 'flip']:
        nodes = list(G.nodes())
        added_edges = 0
        while added_edges < num_edges_to_change:
            u, v = random.sample(nodes, 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                added_edges += 1
                
    # Reconstruct pyg_data
    new_data = from_networkx(G)
    new_data.x = pyg_data.x if hasattr(pyg_data, 'x') else None
    new_data.y = pyg_data.y if hasattr(pyg_data, 'y') else None
    return new_data

def perturb_cycles(pyg_data, cycle_length=5):
    """
    Finds cycles of length `cycle_length` and drops one edge from them to break the cycle.
    """
    G = to_networkx(pyg_data, to_undirected=True)
    cycles = nx.cycle_basis(G)
    
    edges_to_drop = set()
    for cycle in cycles:
        if len(cycle) == cycle_length:
            # Pick a random edge in this cycle
            u = cycle[0]
            v = cycle[1]
            edges_to_drop.add((u, v))
            
    G.remove_edges_from(list(edges_to_drop))
    
    new_data = from_networkx(G)
    new_data.x = pyg_data.x if hasattr(pyg_data, 'x') else None
    new_data.y = pyg_data.y if hasattr(pyg_data, 'y') else None
    return new_data
