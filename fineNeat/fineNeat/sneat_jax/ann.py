import jax 
import jax.numpy as jnp


# -- ANN Ordering -------------------------------------------------------- -- #

def getNodeOrder(nodeG,connG):
    """Builds connection matrix from genome through topological sorting.

    Args:
    nodeG - (np_array) - node genes
            [3 X nUniqueGenes]
            [0,:] == Node Id
            [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
            [2,:] == Activation function (as int)

    connG - (np_array) - connection genes
            [5 X nUniqueGenes] 
            [0,:] == Innovation Number (unique Id)
            [1,:] == Source Node Id
            [2,:] == Destination Node Id
            [3,:] == Weight Value
            [4,:] == Enabled?  

    Returns:
    Q    - [int]      - sorted node order as indices
    wMat - (np_array) - ordered weight matrix
            [N X N]

    OR

    False, False      - if cycle is found

    Todo:
    * setdiff1d is slow, as all numbers are positive ints is there a
        better way to do with indexing tricks (as in quickINTersect)?
    """
    node = jnp.copy(nodeG)
    conn = jnp.copy(connG)
    
    nIns = len(node[0,node[1,:] == 1]) + len(node[0,node[1,:] == 4])
    nOuts = len(node[0,node[1,:] == 2])

    # Create connection and initial weight matrices
    conn = conn.at[3, conn[4,:]==0].set(jnp.nan)  # JAX immutable array update
    src = conn[1,:].astype(jnp.int32)
    dest = conn[2,:].astype(jnp.int32)

    # Reordering node
    reordered_index = jnp.concatenate([
        node[0, node[1,:] == 1],
        node[0, node[1,:] == 4],
        node[0, node[1,:] == 3],
        node[0, node[1,:] == 2]
    ])

    # Get Edge on reordered nodes 
    src_mask = (src.reshape(-1, 1) == reordered_index.reshape(1, -1))
    dest_mask = (dest.reshape(-1, 1) == reordered_index.reshape(1, -1))
    src = (src_mask @ jnp.arange(len(reordered_index)).reshape(-1, 1)).flatten()
    dest = (dest_mask @ jnp.arange(len(reordered_index)).reshape(-1, 1)).flatten()

    # Create weight matrix
    wMat = jnp.zeros((node.shape[1], node.shape[1]))
    wMat = wMat.at[src, dest].set(conn[3,:])

    # Get connection matrix
    connMat = wMat[nIns+nOuts:, nIns+nOuts:]
    connMat = jnp.where(connMat != 0, 1, 0) 

    # Same till here

    # Topological Sort
    edge_in = jnp.sum(connMat, axis=0)
    Q = jnp.where(edge_in == 0)[0]
    for i in range(len(connMat)):
        if (len(Q) == 0) or (i >= len(Q)):
            Q = []
            print("Cycle found, can't sort") 
            return False, False   
        
        edge_out = connMat[Q[i],:]
        edge_in = edge_in - edge_out  # Remove previous layer nodes' conns from total
        
        # Convert numpy operations to JAX
        zero_indices = jnp.where(edge_in == 0)[0]
        nextNodes = jnp.setdiff1d(zero_indices, Q)  # Exclude previous layer nodes
        Q = jnp.concatenate([Q, nextNodes])  # JAX uses concatenate instead of hstack
        
        if jnp.sum(edge_in) == 0:
            break

    # Add inputs and outputs back
    Q = Q + nIns + nOuts
    Q = jnp.concatenate([
        jnp.arange(nIns),
        Q,
        jnp.arange(nIns, nIns+nOuts)
    ])

    return Q, wMat    


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

    wMat = jnp.where(jnp.isnan(wMat), 0, wMat)
    nNode = wMat.shape[0]

    # Create adjacency matrix (1 where connection exists)
    adj = (wMat != 0).astype(jnp.int32)

    # Find nodes with no incoming connections (sources)
    in_degree = jnp.sum(adj, axis=0)
    sources = jnp.where(in_degree == 0)[0]

    # Initialize layers
    layers = jnp.full(nNode, -1)
    layers = layers.at[sources].set(0)

    # Use BFS to assign layers
    current_layer = 0
    iteration = 0
    timeout = 1000
    while True:
        # Check timeout
        iteration += 1
        if iteration > timeout:
            # print("Timeout reached")
            return None
            
        # Find nodes that receive input only from already-assigned layers
        unassigned_mask = (layers == -1)
        if not jnp.any(unassigned_mask):
            break
            
        # Find nodes whose inputs are all from previous layers
        inputs_assigned = ~jnp.any(adj[unassigned_mask], axis=0)
        next_layer = jnp.where(unassigned_mask & inputs_assigned)[0]

        if len(next_layer) == 0:
            break
            
        current_layer += 1
        layers = layers.at[next_layer].set(current_layer)
        
    return layers


def getNodeKey(nodeG, connG): 
    """ 
    Ordered node -- layer index
    """
    nIns = len(nodeG[0,nodeG[1,:] == 1]) + len(nodeG[0,nodeG[1,:] == 4])
    nOuts = len(nodeG[0,nodeG[1,:] == 2])
    order, wMat = getNodeOrder(nodeG, connG)
    if order is False: 
        return False 

    hMat = wMat[nIns:-nOuts,nIns:-nOuts]
    hLay = getLayer(hMat)+1

    if len(hLay) > 0:
        lastLayer = jnp.max(hLay)+1
    else:
        lastLayer = 1
        
    L = jnp.concatenate([
        jnp.zeros(nIns), 
        hLay, 
        jnp.full((nOuts,), lastLayer)
    ])
    nodeKey = jnp.column_stack([nodeG[0,order], L])
    
    return nodeKey


def getNodeMap(nodeG, connG): 
  """ 
  node id --> layer index & order index
  """
  nIns = len(nodeG[0,nodeG[1,:] == 1]) + len(nodeG[0,nodeG[1,:] == 4])
  nOuts = len(nodeG[0,nodeG[1,:] == 2])
  order, wMat = getNodeOrder(nodeG, connG)
  if order is False: 
    return False 

  hMat = wMat[nIns:-nOuts,nIns:-nOuts]
  hLay = getLayer(hMat)+1

  if len(hLay) > 0:
      lastLayer = max(hLay)+1
  else:
      lastLayer = 1
      
  L = jnp.hstack([jnp.zeros(nIns), hLay, jnp.full((nOuts,), lastLayer)]).astype(jnp.int32)
  nodeMap = {}
  for i in range(len(nodeG[0])): 
      nodeMap[int(order[i])] = [L[i].item(), i] # layer index, order index
    
  return nodeMap



def getNodeInfo(nodeG, connG, timeout=50): 
  """ 
  node id --> layer index & order index
  """
  nIns = len(nodeG[0,nodeG[1,:] == 1]) + len(nodeG[0,nodeG[1,:] == 4])
  nOuts = len(nodeG[0,nodeG[1,:] == 2])
  order, wMat = getNodeOrder(nodeG, connG)
  if order is False: 
    return False, False, False

  hMat = wMat[nIns:-nOuts,nIns:-nOuts]
  temp_layer = getLayer(hMat, timeout=timeout)
  if temp_layer is None: 
    return False, order, wMat 
  hLay = temp_layer+1 # need to add timeout for this function

  if len(hLay) > 0:
      lastLayer = max(hLay)+1
  else:
      lastLayer = 1
      
  L = jnp.hstack([jnp.zeros(nIns), hLay, jnp.full((nOuts,), lastLayer)]).astype(jnp.int32)

  nodeMap = {}
  for i in range(len(nodeG[0])): 
      nodeMap[int(order[i])] = [L[i].item(), i] # layer index, order index
    
  return nodeMap, order, wMat



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