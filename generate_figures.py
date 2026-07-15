import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools

# Set overall style
plt.style.use('seaborn-v0_8-white')

# ==========================================
# 1. Cellular Lifting Diagram
# ==========================================
def generate_cellular_lifting():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create hexagon
    G = nx.cycle_graph(6)
    pos = nx.circular_layout(G)
    
    # Left Side: 1-WL view
    ax = axes[0]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='white', node_size=300, edgecolors='black')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', width=2)
    ax.set_title('1-Dimensional Graph\n(1-WL view)', fontsize=16)
    ax.axis('off')
    
    # Right Side: 2-Cell view
    ax = axes[1]
    # Draw filled polygon for the 2-cell
    polygon = patches.Polygon([pos[i] for i in range(6)], closed=True, 
                              facecolor='orange', alpha=0.5, edgecolor='none')
    ax.add_patch(polygon)
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='white', node_size=300, edgecolors='black')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', width=2)
    ax.set_title('2-Cell (Chordless Cycle)', fontsize=16)
    ax.axis('off')
    
    fig.text(0.5, 0.55, 'Topological Lift\n(Co-boundary Operator $B_2$)', 
             ha='center', va='center', fontsize=14, weight='bold')
    fig.text(0.5, 0.45, r'$\longrightarrow$', 
             ha='center', va='center', fontsize=40)
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig('cellular_lifting.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 2. Bottleneck / Over-Squashing Concept
# ==========================================
def generate_bottleneck():
    # Two clusters of 8 nodes
    G1 = nx.complete_graph(8)
    G2 = nx.complete_graph(8)
    
    G = nx.disjoint_union(G1, G2)
    # Add bridge
    G.add_edge(0, 8)
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    # Adjust layout to pull clusters apart
    for i in range(8):
        pos[i][0] -= 1.5
    for i in range(8, 16):
        pos[i][0] += 1.5
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Edges
    edge_colors = []
    edge_widths = []
    
    for u, v in G.edges():
        if (u < 8 and v >= 8) or (u >= 8 and v < 8):
            # Bridge - Highly negative curvature
            edge_colors.append('red')
            edge_widths.append(5.0)
        else:
            # Cluster - Positive curvature
            edge_colors.append('blue')
            edge_widths.append(1.0)
            
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgray', node_size=150, edgecolors='black')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.8)
    
    # Custom legend
    import matplotlib.lines as mlines
    blue_line = mlines.Line2D([], [], color='blue', linewidth=1, label='Positive Curvature (Redundancy)')
    red_line = mlines.Line2D([], [], color='red', linewidth=5, label='Negative Curvature (Bottleneck)')
    ax.legend(handles=[blue_line, red_line], loc='upper center', fontsize=12)
    
    ax.set_title('Forman-Ricci Curvature: Identifying a Structural Bottleneck', fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('bottleneck_oversquashing.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 3. Shrikhande vs. Rook's Graph
# ==========================================
def find_chordless_cycle(G, length):
    for nodes in itertools.combinations(G.nodes(), length):
        subG = G.subgraph(nodes)
        if subG.number_of_edges() == length:
            if nx.is_connected(subG) and all(d == 2 for n, d in subG.degree()):
                cycle_edges = list(nx.find_cycle(subG))
                cycle_nodes = [e[0] for e in cycle_edges]
                return cycle_nodes, cycle_edges
    return None, None

def generate_sr_graphs():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ---------------------------
    # Rook's Graph (4x4 Grid)
    # ---------------------------
    R = nx.Graph()
    for i in range(4):
        for j in range(4):
            R.add_node((i, j))
            
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if k != j:
                    R.add_edge((i, j), (i, k))
                if k != i:
                    R.add_edge((i, j), (k, j))
                    
    pos_R = {(i, j): (j, -i) for i, j in R.nodes()}
    
    ax = axes[0]
    nx.draw_networkx_nodes(R, pos_R, ax=ax, node_color='lightblue', node_size=400, edgecolors='black')
    nx.draw_networkx_edges(R, pos_R, ax=ax, edge_color='lightgray', alpha=0.7, width=1.5)
    
    # Highlight a 3-clique (triangle)
    triangle_nodes = [(0,0), (0,1), (0,2)]
    triangle_edges = [((0,0), (0,1)), ((0,1), (0,2)), ((0,2), (0,0))]
    nx.draw_networkx_nodes(R, pos_R, nodelist=triangle_nodes, ax=ax, node_color='orange', node_size=500, edgecolors='black')
    nx.draw_networkx_edges(R, pos_R, edgelist=triangle_edges, ax=ax, edge_color='orange', width=4)
    
    ax.set_title("Rook's Graph\nLocal neighborhoods have dense triangles (3-cliques)", fontsize=16)
    ax.axis('off')
    
    # ---------------------------
    # Shrikhande Graph
    # ---------------------------
    S = nx.Graph()
    for i in range(4):
        for j in range(4):
            S.add_node((i, j))
            
    diffs = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1)]
    for i in range(4):
        for j in range(4):
            for di, dj in diffs:
                ni = (i + di) % 4
                nj = (j + dj) % 4
                S.add_edge((i, j), (ni, nj))
                
    pos_S = nx.circular_layout(S)
    
    ax = axes[1]
    nx.draw_networkx_nodes(S, pos_S, ax=ax, node_color='lightgreen', node_size=400, edgecolors='black')
    nx.draw_networkx_edges(S, pos_S, ax=ax, edge_color='lightgray', alpha=0.7, width=1.5)
    
    # Highlight a hexagon
    hex_nodes, hex_edges = find_chordless_cycle(S, 6)
    if not hex_nodes:
        # Fallback to a cycle of 4 or something if 6 doesn't exist
        hex_nodes, hex_edges = find_chordless_cycle(S, 4)

    if hex_nodes:
        nx.draw_networkx_nodes(S, pos_S, nodelist=hex_nodes, ax=ax, node_color='magenta', node_size=500, edgecolors='black')
        nx.draw_networkx_edges(S, pos_S, edgelist=hex_edges, ax=ax, edge_color='magenta', width=4)
    
    ax.set_title("Shrikhande Graph\nFundamental cycles are distinct (e.g., hexagons)", fontsize=16)
    ax.axis('off')
    
    # Punchline caption
    fig.text(0.5, 0.05, "Both graphs have identical local degree distributions (confusing 1-WL test),\nbut their 2-cell mesoscale structures are entirely distinct.", 
             ha='center', va='center', fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig('shrikhande_vs_rooks.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    generate_cellular_lifting()
    generate_bottleneck()
    generate_sr_graphs()
    print("Images generated successfully!")
