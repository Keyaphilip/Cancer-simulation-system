import networkx as nx
import matplotlib.pyplot as plt
from src.pathways import make_example_ddr

def draw_pathway(G, values=None, filename=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    
    # Determine node colors
    if values:
        # Normalize values for color mapping (e.g., -1 to 1)
        node_colors = [values.get(n, 0.0) for n in G.nodes()]
        cmap = plt.cm.coolwarm
        vmin, vmax = -1, 1
    else:
        node_colors = 'lightblue'
        cmap = None
        vmin, vmax = None, None

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    # Draw edges based on sign
    edges = G.edges(data=True)
    positive_edges = [(u, v) for u, v, d in edges if d.get('sign') == 1]
    negative_edges = [(u, v) for u, v, d in edges if d.get('sign') == -1]
    
    nx.draw_networkx_edges(G, pos, edgelist=positive_edges, edge_color='green', arrows=True, width=2, arrowstyle='-|>', arrowsize=20, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=negative_edges, edge_color='red', arrows=True, width=2, arrowstyle='-[', arrowsize=20, ax=ax)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Activation (+1)'),
        Line2D([0], [0], color='red', lw=2, label='Inhibition (-1)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title("DNA Damage Response Pathway")
    ax.axis('off')
    
    if filename:
        plt.savefig(filename)
        print(f"Graph saved to {filename}")
    
    return fig

if __name__ == "__main__":
    G = make_example_ddr()
    draw_pathway(G)
