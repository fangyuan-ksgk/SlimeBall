from ..neat_src.ann import getNodeInfo 
import networkx as nx 
import numpy as np 
import matplotlib.pyplot as plt 
import io
from PIL import Image

def visualize_dag(wMat, seq2order, seq2node, seq2layer, nIns, nOuts, figsize=(10, 10)):
    """
    Visualize neural network as a DAG using networkx
    
    Args:
        wMat: Weight matrix (2D numpy array)
        seq2order: Dictionary mapping sequence index to order
        seq2node: Dictionary mapping sequence index to node index
        seq2layer: Dictionary mapping sequence index to layer
        figsize: Tuple for figure size
    """    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    nodelabels = []
    for seq_idx in seq2node.keys():
        if seq_idx >= nIns and seq2layer[seq_idx] == 0: 
            continue
        G.add_node(seq_idx, 
                  node_id=seq2node[seq_idx],
                  layer=seq2layer[seq_idx],
                  order=seq2order[seq_idx])
        label = f"Input {seq_idx + 1}" if seq_idx < nIns-1 else "Bias" if seq_idx == nIns-1 else f"Output {seq_idx - len(seq2order) + nOuts + 1}" if seq_idx >= len(seq2order) - nOuts else ""
        nodelabels.append(label)
    
    # Add edges with weights from wMat
    rows, cols = np.where((wMat != 0) & ~np.isnan(wMat))
    for row, col in zip(rows, cols):
        weight = wMat[row, col]
        G.add_edge(row, col, weight=weight)
    
    # Calculate node positions
    # Group nodes by layer
    layers = {}
    for node in G.nodes():
        layer = seq2layer[node]
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)
    
    # Position nodes
    pos = {}
    fig_wide = 10
    fig_height = 5
    
    # Assign x coordinates by layer
    max_layer = max(layers.keys())
    max_layer_nodes = max([len(nodes) for nodes in layers.values()])
    for layer_idx, nodes in layers.items():
        # Sort nodes within layer by order
        nodes.sort(key=lambda x: seq2order[x])
        
        # Calculate x position normalized to fig_wide
        x = (layer_idx/max_layer) * fig_wide
        
        # Calculate y positions for nodes in this layer
        layer_nodes = len(nodes)
        width_ratio = min(layer_nodes/max_layer_nodes, 0.6)
        layer_width = fig_height * width_ratio
        
        y_coords = np.linspace(fig_height/2+layer_width/2, fig_height/2-layer_width/2, layer_nodes)
            
        for node, y in zip(nodes, y_coords):
            pos[node] = (x, y)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                          node_color='lightblue',
                          node_size=800,
                          node_shape='o',
                          alpha=0.8)
    
    # Draw input labels with arrows
    for i in range(nIns):
        if i in pos:  # Check if node exists in position dictionary
            plt.annotate(nodelabels[i], 
                        xy=(pos[i][0]-0.2, pos[i][1]),
                        xytext=(pos[i][0] - 2, pos[i][1]),
                        arrowprops=dict(arrowstyle="->", color='green', lw=1),
                        ha='right',
                        va='center')
    for i in range(nOuts):
        plt.annotate(nodelabels[-i-1], 
                    xy=(pos[len(seq2order)-nOuts+i][0]+0.2, pos[len(seq2order)-nOuts+i][1]),
                    xytext=(pos[len(seq2order)-nOuts+i][0] + 2, pos[len(seq2order)-nOuts+i][1]),
                    arrowprops=dict(arrowstyle="<-", color='red', lw=1),
                    ha='left',
                    va='center')
    
    # Draw edges with width proportional to weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Normalize weights to [0,1] range for alpha
    alphas = np.abs(weights) / np.max(np.abs(weights) + 0.1)
        
    # Create color list based on weight signs
    colors = ['lightblue' if w > 0 else '#ffb3b3' for w in weights]
    
    # Draw edges with individual alpha and color values
    for (edge, alpha, color) in zip(G.edges(), alphas, colors):
        nx.draw_networkx_edges(G, pos, edgelist=[edge],
            alpha=float(alpha),  # Convert to float in case of numpy type
            width=1.0,
            edge_color=[color],
            arrowsize=8)
    
    # Add labels
    labels = {node: f"{seq2node[node]}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels)
    
    plt.title("Neural Network DAG Visualization")
    plt.axis('off')
    
    return plt.gcf(), plt.gca()


def viewInd(ind, figsize=(10, 10)): 
    # Create Graph
    nIn = ind.nInput + ind.nBias # fixed 
    nOut= ind.nOutput
    node_map, seq_node_indices, wMat = getNodeInfo(ind.node, ind.conn)
    layer = np.array([node_map[node_idx][0] for node_idx in seq_node_indices])

    seq2node = {seq_idx: node_idx for seq_idx, node_idx in enumerate(seq_node_indices)}
    seq2order = {seq_idx: node_map[seq2node[seq_idx]][1] for seq_idx in range(len(node_map))}
    seq2layer = {seq_idx: node_map[seq2node[seq_idx]][0] for seq_idx in range(len(node_map))}

    order2seq = {seq2order[seq_idx]: seq_idx for seq_idx in range(len(seq2order))}
    order2layer = {order_idx: seq2layer[order2seq[order_idx]] for order_idx in range(len(order2seq))}
    
    return visualize_dag(wMat, seq2order, seq2node, seq2layer, nIn, nOut, figsize=figsize)

def cLinspace(start,end,N):
  if N == 1:
    return np.mean([start,end])
  else:
    return np.linspace(start,end,N)

def lload(fileName):
  return np.loadtxt(fileName, delimiter=',') 

def fig2img(fig):
    # Save figure to a temporary buffer.
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return Image.open(buf)


def draw_img(ind): 
    fig, ax = viewInd(ind)
    
    ax.text(0.7, 0.9, f"Active Connections: {ind.nConns()}\nNumber of Layers: {ind.max_layer}",
        bbox=dict(facecolor='white', edgecolor='black', pad=10),
        horizontalalignment='center', fontsize=16,
        transform=ax.transAxes)
    
    img = fig2img(fig)
    plt.close(fig)
    return img