import jax 
import jax.numpy as jnp


def getMatOrder(nIns, nOuts, wMat):
    """Get topological order ensuring inputs come first"""
    n = len(wMat)
    order = []
    processed = set()
    
    # Calculate in-degree for each node (excluding input nodes)
    in_degree = jnp.zeros(n)
    for j in range(nIns, n):
        in_degree = in_degree.at[j].set(
            jnp.sum((wMat[:, j] != 0) & (~jnp.isnan(wMat[:, j])))
        )
    
    # Start with input nodes
    for i in range(nIns):
        order.append(i)
        processed.add(i)
    
    # Create priority queue of non-input nodes sorted by in-degree
    candidate_nodes = sorted(
        [(node, in_degree[node].item()) for node in range(nIns, n)],
        key=lambda x: x[1]  # Sort by in-degree
    )
    
    # Process nodes in order of increasing in-degree
    idx = 0
    while idx < len(candidate_nodes) and len(order) < n:
        node, _ = candidate_nodes[idx]
        
        if node not in processed:
            # Check if all incoming connections are from processed nodes
            incoming = jnp.where((wMat[:, node] != 0) & (~jnp.isnan(wMat[:, node])))[0]
            if all(int(i) in processed for i in incoming):
                order.append(node)
                processed.add(node)
                # Reset to start of queue as new nodes might now be processable
                idx = 0
                continue
                
        idx += 1
    
    # Check for cycles
    if len(order) != n:
        unprocessed = set(range(n)) - processed
        print(f" :: Cycle detected in neural network. Unprocessed nodes: {unprocessed}")
        return False
        
    return jnp.array(order)
  
  
  
def calwMat(node, conn): 
    # Set nan values for disabled connections
    conn = conn.at[3, conn[4,:]==0].set(jnp.nan)
    src  = conn[1,:].astype(jnp.int32)
    dest = conn[2,:].astype(jnp.int32)

    # wMat having order: input, bias, hidden, output -- important for propagation
    seq2node = jnp.concatenate([
        node[0, node[1,:]==1], 
        node[0, node[1,:]==4], 
        node[0, node[1,:]==3], 
        node[0, node[1,:]==2]
    ])
    seq2node = seq2node.astype(jnp.int32)

    src_mask = (src.reshape(-1, 1) == seq2node.reshape(1, -1))  # (n_conn, n_node)
    dest_mask = (dest.reshape(-1, 1) == seq2node.reshape(1, -1))
    src = (src_mask @ jnp.arange(len(seq2node)).reshape(-1, 1)).flatten()  # Convert to 1D
    dest = (dest_mask @ jnp.arange(len(seq2node)).reshape(-1, 1)).flatten()  # Convert to 1D

    # Create weight matrix according to reordered nodes
    wMat = jnp.zeros((node.shape[1], node.shape[1]))
    wMat = wMat.at[src, dest].set(conn[3,:])  # assign weight to the connection
    
    return wMat, seq2node
  
  
  
def getMat(nodeG, connG): 
    """ 
    Get Connection Weight Matrix for reordered Nodes
    """
    node = jnp.array(nodeG)
    conn = jnp.array(connG)
    nIns = len(node[0,node[1,:] == 1]) + len(node[0,node[1,:] == 4])
    nOuts = len(node[0,node[1,:] == 2])

    wMat, seq2node = calwMat(node, conn)
    
    order2seq = getMatOrder(nIns, nOuts, wMat)
    
    if order2seq is False:
        return False, False, False, False
    
    seq2order = {int(order2seq[int(seq_idx)]): int(seq_idx) for seq_idx in order2seq}
    node2seq = {int(node_id): int(seq_idx) for node_id, seq_idx in zip(seq2node, jnp.arange(len(seq2node)))}
    seq2node = {int(node2seq[int(node_id)]): int(node_id) for node_id in node2seq}
    node2order = {int(node_idx): int(seq2order[int(node2seq[int(node_idx)])]) for node_idx in node2seq}
    return wMat, node2order, node2seq, seq2node
  
  
def getLayer(wMat, node2seq, node2order, seq2node):
    order2node = {node2order[node_idx]: node_idx for node_idx in node2order}
    node2layer = {}
    for order_idx in range(len(order2node)): 
        node_idx = order2node[order_idx]
        # Find all nodes that connect to this node
        row_idx = node2seq[node_idx]
        input_node_seq_ids = jnp.where((wMat[:, row_idx] != 0) & (~jnp.isnan(wMat[:, row_idx])))[0]
        input_node_ids = [seq2node[int(i)] for i in input_node_seq_ids]
        if len(input_node_ids) == 0:
            # Input nodes (no incoming connections)
            node2layer[node_idx] = 0
        else:
            # Node's layer is max layer of inputs + 1
            input_node_layers = [node2layer[i] for i in input_node_ids]
            node2layer[node_idx] = jnp.max(jnp.array(input_node_layers)) + 1
    return node2layer 


def getNodeInfo(nodeG, connG): 
    wMat, node2order, node2seq, seq2node = getMat(nodeG, connG)
    if wMat is False:
        return False, False, False
    node2layer = getLayer(wMat, node2seq, node2order, seq2node)
    nodemap = {int(node_idx): (node2layer[int(node_idx)], node2order[int(node_idx)]) for node_idx in nodeG[0]}
    seq_node_indices = [seq2node[seq_idx] for seq_idx in range(len(seq2node))]
    return nodemap, seq_node_indices, wMat


def applyAct(actId, x):
  """Returns value after an activation function is applied
  Lookup table to allow activations to be stored in numpy arrays

  case 1  -- Linear
  case 2  -- Unsigned Step Function
  case 3  -- Sin
  case 4  -- Gausian with mean 0 and sigma 1
  case 5  -- Hyperbolic Tangent [tanh] (signed)
  case 6  -- Sigmoid unsigned [1 / (1 + exp(-x))]
  case 7  -- Inverse
  case 8  -- Absolute Value
  case 9  -- Relu
  case 10 -- Cosine
  case 11 -- Squared

  Args:
    actId   - (int)   - key to look up table
    x       - (???)   - value to be input into activation
              [? X ?] - any type or dimensionality

  Returns:
    output  - (float) - value after activation is applied
              [? X ?] - same dimensionality as input
  """
  if actId == 1:   # Linear
    value = x

  if actId == 2:   # Unsigned Step Function
    value = 1.0*(x>0.0)
    #value = (jnp.tanh(50*x/2.0) + 1.0)/2.0

  elif actId == 3: # Sin
    value = jnp.sin(jnp.pi*x) 

  elif actId == 4: # Gaussian with mean 0 and sigma 1
    value = jnp.exp(-jnp.multiply(x, x) / 2.0)

  elif actId == 5: # Hyperbolic Tangent (signed)
    value = jnp.tanh(x)     

  elif actId == 6: # Sigmoid (unsigned)
    value = (jnp.tanh(x/2.0) + 1.0)/2.0

  elif actId == 7: # Inverse
    value = -x

  elif actId == 8: # Absolute Value
    value = jnp.abs(x)   
    
  elif actId == 9: # Relu
    value = jnp.maximum(0, x)   

  elif actId == 10: # Cosine
    value = jnp.cos(jnp.pi*x)

  elif actId == 11: # Squared
    value = x**2
    
  else:
    value = x

  return value


def act(weights, aVec, nInput, nOutput, inPattern):
  """Returns FFANN output given a single input pattern
  If the variable weights is a vector it is turned into a square weight matrix.
  
  Allows the network to return the result of several samples at once if given a matrix instead of a vector of inputs:
      Dim 0 : individual samples
      Dim 1 : dimensionality of pattern (# of inputs)

  Args:
    weights   - (np_array) - ordered weight matrix or vector
                [N X N] or [N**2]
    aVec      - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
    nInput    - (int)      - number of input nodes
    nOutput   - (int)      - number of output nodes
    inPattern - (np_array) - input activation
                [1 X nInput] or [nSamples X nInput]

  Returns:
    output    - (np_array) - output activation
                [1 X nOutput] or [nSamples X nOutput]
  """
  # Turn weight vector into weight matrix
  if jnp.ndim(weights) < 2:
      nNodes = int(jnp.sqrt(jnp.shape(weights)[0]))
      wMat = jnp.reshape(weights, (nNodes, nNodes))
  else:
      nNodes = jnp.shape(weights)[0]
      wMat = weights
  wMat = jnp.where(jnp.isnan(wMat), 0, wMat)

  # Vectorize input
  if jnp.ndim(inPattern) > 1:
      nSamples = jnp.shape(inPattern)[0]
  else:
      nSamples = 1

  # Run input pattern through ANN    
  nodeAct  = jnp.zeros((nSamples,nNodes)) # Store activation of each node
  nodeAct = nodeAct.at[:,0].set(1) # Bias activation
  nodeAct = nodeAct.at[:,1:nInput+1].set(inPattern) # Prepare input node activation

  # Propagate signal through hidden to output nodes
  for iNode in range(nInput+1,nNodes):
      rawAct = jnp.dot(nodeAct, wMat[:,iNode]).squeeze()
      nodeAct = nodeAct.at[:,iNode].set(applyAct(aVec[iNode], rawAct)) # Looping sparse dot-product to compute each node's activation
  output = nodeAct[:,-nOutput:]   
  return output


def obtainOutgoingConnections(connG, node_id): 
  if connG is not None:
    srcIndx = jnp.where(connG[1,:]==node_id)[0]
    exist = connG[2,srcIndx]
    exist = jnp.unique(exist).astype(jnp.int32)
    return exist
  else:
    return []


class NeatPolicy: 
    def __init__(self, indiv, game): 
        self.indiv = indiv 
        self.game = game 
        
        if self.indiv.aVec is None or self.indiv.aVec is not None and not isinstance(self.indiv.aVec, jnp.ndarray): 
            self.indiv.express()
        self.indiv.aVec[-1] = 1
    
    def predict(self, input): 
        return act(self.indiv.wMat, self.indiv.aVec, self.game.input_size, self.game.output_size, input)[0]


class NEATPolicy: 
  def __init__(self, json_path): 
      from .ind import Ind 
      self.indiv = Ind.load(json_path)

      if self.indiv.aVec is None or self.indiv.aVec is not None and not isinstance(self.indiv.aVec, jnp.ndarray): 
          self.indiv.express()
      self.indiv.aVec[-1] = 1
    
      
  def predict(self, input): 
      return act(self.indiv.wMat, self.indiv.aVec, self.indiv.nInput, self.indiv.nOutput, input)[0]