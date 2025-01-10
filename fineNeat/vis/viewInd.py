from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import sys
sys.path.append('../domain/')
sys.path.append('vis')
from domain.config import games
from neat_src.ind import getNodeOrder, getLayer, getNodeMap

def get_nodeMap(ind): 
  """ 
  NodeMap: order_id, node_id (index on graph)
  """
  nodeMap = getNodeMap(ind.node, ind.conn)
  nodeMap = {nodeMap[id][1]:id for id in nodeMap} # order -> node id map
  return nodeMap

def viewInd(ind, taskName):
  env = games[taskName]
  if isinstance(ind, str):
    ind = np.loadtxt(ind, delimiter=',') 
    wMat = ind[:,:-1]
    aVec = ind[:,-1]
  else:
    wMat = ind.wMat
    aVec = np.zeros((np.shape(wMat)[0]))  
  print('# of Connections in ANN: ', np.sum(wMat!=0))
    
  # Create Graph
  nIn = ind.nInput + ind.nBias # fixed 
  nOut= ind.nOutput
  G, layer= ind2graph(wMat, nIn, nOut) # pass | G is off by one node (likely hidden node is missing?)
  pos = getNodeCoord(G,layer,nIn, nOut)
    
  # Draw Graph
  fig = plt.figure(figsize=(10,10), dpi=100)
  ax = fig.add_subplot(111)
  drawEdge(G, pos, wMat, layer)
  nx.draw_networkx_nodes(G, pos,
      node_color='lightblue',
      node_size=800,           # Increased from default
      node_shape='o')
  nodeMap = get_nodeMap(ind)
  drawNodeLabels(G,pos,aVec, nodeMap) 
  labelInOut(pos,ind)
  
  # Add margins to prevent cutoff
  ax.margins(0.2)
  
  plt.tick_params(
      axis='both',
      which='both',
      bottom=False,
      top=False,
      left=False,
      labelleft=False,
      labelbottom=False)
    
  return fig, ax


def ind2graph(wMat, nIn, nOut):
    hMat = wMat[nIn:-nOut,nIn:-nOut]
    hLay = getLayer(hMat)+1

    if len(hLay) > 0:
      lastLayer = max(hLay)+1
    else:
      lastLayer = 1
    L = np.r_[np.zeros(nIn), hLay, np.full((nOut),lastLayer) ]

    layer = L
    order = layer.argsort()
    layer = layer[order]

    wMat = wMat[np.ix_(order,order)]
    nLayer = layer[-1]

    # Convert wMat to Full Network Graph
    rows, cols = np.where(wMat != 0)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G, layer


def getNodeCoord(G,layer,nIn, nOut):
  
    # Calculate positions of input and output
    nNode= len(G.nodes) # wrong value
    fixed_pos = np.empty((nNode,2))
    fixed_nodes = np.r_[np.arange(0,nIn),np.arange(nNode-nOut,nNode)]

    # Set Figure dimensions
    fig_wide = 10
    fig_long = 5

    # Assign x and y coordinates per layer
    x = np.ones((1,nNode))*layer # Assign x coord by layer
    x = (x/np.max(x))*fig_wide # Normalize

    _, nPerLayer = np.unique(layer, return_counts=True)

    y = cLinspace(fig_long+2, -2, nPerLayer[0])
    for i in range(1,len(nPerLayer)):
      if i%2 == 0:
        y = np.r_[y,cLinspace(fig_long, 0, nPerLayer[i])]
      else:
        y = np.r_[y,cLinspace(fig_long+1, -1, nPerLayer[i])]

    fixed_pos = np.c_[x.T,y.T]
    pos = dict(enumerate(fixed_pos.tolist()))
    
    return pos
  
def labelInOut(pos, ind, env=None):
    """Label nodes in network visualization based on Ind structure
    
    Args:
        pos: Node position dictionary for visualization
        ind: Individual containing network structure
        env: Optional environment for custom labels
    """
    nNode = len(pos)
    
    # Create default labels following specified order
    stateLabels = (
        [f"Input {i+1}" for i in range(ind.nInput)] +      # Input nodes
        [f"Bias" for i in range(ind.nBias)] +     # Bias nodes
        [f"Hidden" for i in range(ind.nHidden)] +    # Hidden nodes
        [f"Output {i+1}" for i in range(ind.nOutput)]      # Output nodes
    )
    
    # Override with environment labels if available
    if env and hasattr(env, 'in_out_labels') and len(env.in_out_labels) > 0:
        input_labels = env.in_out_labels[:ind.nInput]
        output_labels = env.in_out_labels[-ind.nOutput:]
        
        # Replace default labels while preserving structure
        stateLabels[:ind.nInput] = input_labels
        stateLabels[-ind.nOutput:] = output_labels
    
    # Create label dictionary
    labelDict = {i: label for i, label in enumerate(stateLabels)}
    
    # Draw input and bias labels
    for i in range(ind.nInput + ind.nBias):
        plt.annotate(labelDict[i], 
                    xy=(pos[i][0]-0.5, pos[i][1]), 
                    xytext=(pos[i][0]-2.5, pos[i][1]-0.5),
                    arrowprops=dict(arrowstyle="->", color='k', connectionstyle="angle"))
    
    # Draw output labels
    for i in range(nNode - ind.nOutput, nNode):
        plt.annotate(labelDict[i], 
                    xy=(pos[i][0]+0.1, pos[i][1]), 
                    xytext=(pos[i][0]+1.5, pos[i][1]+1.0),
                    arrowprops=dict(arrowstyle="<-", color='k', connectionstyle="angle"))
    
def drawNodeLabels(G, pos, aVec, nodeMap):  
    actLabel = np.array((['','( + )','(0/1)','(sin)','(gau)','(tanh)',\
                         '(sig)','( - )', '(abs)','(relu)','(cos)']))
    listLabel = actLabel[aVec.astype(int)]
    
    # Create labels using nodes from G
    label = {node: f"{nodeMap[node]}\n{listLabel[node]}" for node in G.nodes()}
    
    # Create a new position dict with shifted y coordinates
    pos_attrs = {node: (coord[0], coord[1] - 0.2) for node, coord in pos.items()}  # Adjust -0.1 to shift more/less
    
    nx.draw_networkx_labels(G, pos_attrs, labels=label)
  
  
def drawEdge(G, pos, wMat, layer):
    wMat[np.isnan(wMat)]=0
    # Organize edges by layer
    _, nPerLayer = np.unique(layer, return_counts=True)
    edgeLayer = []
    layBord = np.cumsum(nPerLayer)
    for i in range(0,len(layBord)):
      tmpMat = np.copy(wMat)
      start = layBord[-i]
      end = layBord[-i+1]
      tmpMat[:,:start] *= 0
      tmpMat[:,end:] *= 0
      rows, cols = np.where(tmpMat != 0)
      
      # Skip if no edges in this layer
      if len(rows) == 0:
          continue
          
      weights = tmpMat[rows, cols]  # Get weights for each edge
      edges = list(zip(rows.tolist(), cols.tolist()))
      edgeLayer.append((nx.DiGraph(), weights))  # Store weights with graph
      edgeLayer[-1][0].add_edges_from(edges)
    
    # Handle case where first layer has edges
    if edgeLayer:
        edgeLayer.append(edgeLayer.pop(0))

    # Draw edges with weight-based properties
    for graph, weights in edgeLayer:
        if len(weights) == 0:  # Skip empty layers
            continue
            
        # Normalize weights to [0,1] range for alpha
        alphas = np.abs(weights) / max(1.0, np.max(np.abs(weights)))
        # Create color list based on weight signs (lightblue for positive, red for negative)
        colors = ['lightblue' if w > 0 else '#ffb3b3' for w in weights]
        
        # Draw edges with individual alpha and color values
        for (edge, alpha, color) in zip(graph.edges(), alphas, colors):
            nx.draw_networkx_edges(G, pos, edgelist=[edge],
                alpha=float(alpha),  # Convert to float in case of numpy type
                width=1.0,
                edge_color=[color],
                arrowsize=8)


def getLayer(wMat, timeout=1000):
  """Get layer of each node in weight matrix using a more efficient approach.
  Instead of iterating until convergence, we can use a graph traversal approach.

  Args:
    wMat    - (np_array) - ordered weight matrix [N X N]
    timeout - (int)      - maximum number of iterations before timing out

  Returns:
    layer   - [int]      - layer # of each node
             or None if timeout is reached
  """
  wMat[np.isnan(wMat)] = 0
  nNode = wMat.shape[0]
  
  # Create adjacency matrix (1 where connection exists)
  adj = (wMat != 0).astype(int)
  
  # Find nodes with no incoming connections (sources)
  in_degree = adj.sum(axis=0)
  sources = np.where(in_degree == 0)[0]
  
  # Initialize layers
  layers = np.full(nNode, -1)
  layers[sources] = 0
  
  # Use BFS to assign layers
  current_layer = 0
  iteration = 0
  while True:
    # Check timeout
    iteration += 1
    if iteration > timeout:
      return None
      
    # Find nodes that receive input only from already-assigned layers
    unassigned_mask = (layers == -1)
    if not np.any(unassigned_mask):
      break
      
    # Find nodes whose inputs are all from previous layers
    inputs_assigned = ~np.any(adj[unassigned_mask], axis=0)
    next_layer = np.where(unassigned_mask & inputs_assigned)[0]
    
    if len(next_layer) == 0:
      break
      
    current_layer += 1
    layers[next_layer] = current_layer
    
  return layers

def cLinspace(start,end,N):
  if N == 1:
    return np.mean([start,end])
  else:
    return np.linspace(start,end,N)

def lload(fileName):
  return np.loadtxt(fileName, delimiter=',') 



