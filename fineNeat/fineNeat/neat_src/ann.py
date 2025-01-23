import numpy as np


def getMatOrder(nIns, nOuts, wMat):
    """Get topological order ensuring inputs come first"""
    n = len(wMat)
    order = []
    processed = set()
    
    # Calculate in-degree for each node (excluding input nodes)
    in_degree = np.zeros(n)
    for j in range(nIns, n):
        in_degree[j] = np.sum((wMat[:, j] != 0) & (~np.isnan(wMat[:, j])))
    
    # Start with input nodes
    for i in range(nIns):
        order.append(i)
        processed.add(i)
    
    # Create priority queue of non-input nodes sorted by in-degree
    candidate_nodes = sorted(
        [(node, in_degree[node]) for node in range(nIns, n)],
        key=lambda x: x[1]  # Sort by in-degree
    )
    
    # Process nodes in order of increasing in-degree
    idx = 0
    while idx < len(candidate_nodes) and len(order) < n:
        node, _ = candidate_nodes[idx]
        
        if node not in processed:
            # Check if all incoming connections are from processed nodes
            incoming = np.where((wMat[:, node] != 0) & (~np.isnan(wMat[:, node])))[0]
            if all(i in processed for i in incoming):
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
        
    return np.array(order)
  
  
def calwMat(node, conn): 
    conn[3,conn[4,:]==0] = np.nan
    src  = conn[1,:].astype(int)
    dest = conn[2,:].astype(int)

    # wMat having order: input, bias, hidden, output -- important for propagation
    seq2node = np.r_[node[0, node[1,:]==1], node[0, node[1,:]==4], node[0, node[1,:]==3], node[0, node[1,:]==2]]
    seq2node = seq2node.astype(int)

    src_mask = (src.reshape(-1, 1) == seq2node.reshape(1, -1)) # (n_conn, n_node)
    dest_mask = (dest.reshape(-1, 1) == seq2node.reshape(1, -1))
    src = (src_mask @ np.arange(len(seq2node)).reshape(-1, 1)).flatten()  # Convert to 1D
    dest = (dest_mask @ np.arange(len(seq2node)).reshape(-1, 1)).flatten()  # Convert to 1D

    # Create weight matrix according to reordered nodes
    wMat = np.zeros((np.shape(node)[1],np.shape(node)[1]))
    wMat[src,dest] = conn[3,:] # assign weight to the connection
    
    return wMat, seq2node

def getMat(nodeG, connG): 
    """ 
    Get Connection Weight Matrix for reordered Nodes
    """
    node = np.copy(nodeG)
    conn = np.copy(connG)
    nIns = len(node[0,node[1,:] == 1]) + len(node[0,node[1,:] == 4])
    nOuts = len(node[0,node[1,:] == 2])

    wMat, seq2node = calwMat(node, conn)
    
    order2seq = getMatOrder(nIns, nOuts, wMat)
    
    if order2seq is False:
        return False, False, False, False
      
    # re-order wMat to topological structure
    wMat = wMat[order2seq, :][:, order2seq]
    order2seq = np.arange(len(order2seq))
  
    seq2order = {order2seq[seq_idx]: seq_idx for seq_idx in order2seq}
    node2seq = {node_id: seq_idx for node_id, seq_idx in zip(seq2node, np.arange(len(seq2node)))}
    seq2node = {node2seq[node_id]: node_id for node_id in node2seq}
    node2order = {node_idx: seq2order[node2seq[node_idx]] for node_idx in node2seq}
    
    return wMat, node2order, node2seq, seq2node 


def getLayer(wMat, node2seq, node2order, seq2node):
    order2node = {node2order[node_idx]: node_idx for node_idx in node2order}
    node2layer = {}
    for order_idx in range(len(order2node)): 
        node_idx = order2node[order_idx]
        # Find all nodes that connect to this node
        row_idx = node2seq[node_idx]
        input_node_seq_ids = np.where((wMat[:, row_idx] != 0) & (~np.isnan(wMat[:, row_idx])))[0]
        input_node_ids = [seq2node[int(i)] for i in input_node_seq_ids]
        if len(input_node_ids) == 0:
            # Input nodes (no incoming connections)
            node2layer[node_idx] = 0
        else:
            # Node's layer is max layer of inputs + 1
            input_node_layers = [node2layer[i] for i in input_node_ids]
            node2layer[node_idx] = np.max(input_node_layers) + 1
    return node2layer 
  

def getNodeInfo(nodeG, connG): 
    wMat, node2order, node2seq, seq2node = getMat(nodeG, connG)
    if wMat is False:
        return False, False, False
    node2layer = getLayer(wMat, node2seq, node2order, seq2node)
    nodemap = {int(node_idx): (node2layer[node_idx], node2order[node_idx]) for node_idx in nodeG[0]}
    seq_node_indices = [seq2node[seq_idx] for seq_idx in range(len(seq2node))]
    return nodemap, seq_node_indices, wMat


# -- ANN Activation ------------------------------------------------------ -- #

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
  if np.ndim(weights) < 2:
      nNodes = int(np.sqrt(np.shape(weights)[0]))
      wMat = np.reshape(weights, (nNodes, nNodes))
  else:
      nNodes = np.shape(weights)[0]
      wMat = weights
  wMat[np.isnan(wMat)]=0

  # Vectorize input
  if np.ndim(inPattern) > 1:
      nSamples = np.shape(inPattern)[0]
  else:
      nSamples = 1

  # Run input pattern through ANN    
  nodeAct  = np.zeros((nSamples,nNodes)) # Store activation of each node
  nodeAct[:,0] = 1 # Bias activation
  nodeAct[:,1:nInput+1] = inPattern # Prepare input node activation

  # Propagate signal through hidden to output nodes
  for iNode in range(nInput+1,nNodes):
      rawAct = np.dot(nodeAct, wMat[:,iNode]).squeeze()
      nodeAct[:,iNode] = applyAct(aVec[iNode], rawAct) # Looping sparse dot-product to compute each node's activation
  output = nodeAct[:,-nOutput:]   
  return output


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
    #value = (np.tanh(50*x/2.0) + 1.0)/2.0

  elif actId == 3: # Sin
    value = np.sin(np.pi*x) 

  elif actId == 4: # Gaussian with mean 0 and sigma 1
    value = np.exp(-np.multiply(x, x) / 2.0)

  elif actId == 5: # Hyperbolic Tangent (signed)
    value = np.tanh(x)     

  elif actId == 6: # Sigmoid (unsigned)
    value = (np.tanh(x/2.0) + 1.0)/2.0

  elif actId == 7: # Inverse
    value = -x

  elif actId == 8: # Absolute Value
    value = abs(x)   
    
  elif actId == 9: # Relu
    value = np.maximum(0, x)   

  elif actId == 10: # Cosine
    value = np.cos(np.pi*x)

  elif actId == 11: # Squared
    value = x**2
    
  else:
    value = x

  return value


# -- Action Selection ---------------------------------------------------- -- #

def selectAct(action, actSelect):  
  """Selects action based on vector of actions

    Single Action:
    - Hard: a single action is chosen based on the highest index
    - Prob: a single action is chosen probablistically with higher values
            more likely to be chosen

    We aren't selecting a single action:
    - Softmax: a softmax normalized distribution of values is returned
    - Default: all actions are returned 

  Args:
    action   - (np_array) - vector weighting each possible action
                [N X 1]

  Returns:
    i         - (int) or (np_array)     - chosen index
                         [N X 1]
  """  
  if actSelect == 'softmax':
    action = softmax(action)
  elif actSelect == 'prob':
    action = weightedRandom(np.sum(action,axis=0))
  else:
    action = action.flatten()
  return action

def softmax(x):
    """Compute softmax values for each sets of scores in x.
    Assumes: [samples x dims]

    Args:
      x - (np_array) - unnormalized values
          [samples x dims]

    Returns:
      softmax - (np_array) - softmax normalized in dim 1
    
    Todo: Untangle all the transposes...    
    """    
    if x.ndim == 1:
      e_x = np.exp(x - np.max(x))
      return e_x / e_x.sum(axis=0)
    else:
      e_x = np.exp(x.T - np.max(x,axis=1))
      return (e_x / e_x.sum(axis=0)).T

def weightedRandom(weights):
  """Returns random index, with each choices chance weighted
  Args:
    weights   - (np_array) - weighting of each choice
                [N X 1]

  Returns:
    i         - (int)      - chosen index
  """
  minVal = np.min(weights)
  weights = weights - minVal # handle negative vals
  cumVal = np.cumsum(weights)
  pick = np.random.uniform(0, cumVal[-1])
  for i in range(len(weights)):
    if cumVal[i] >= pick:
      return i
        

# -- File I/O ------------------------------------------------------------ -- #

def exportNet(filename,wMat, aVec):
  indMat = np.c_[wMat,aVec]
  np.savetxt(filename, indMat, delimiter=',',fmt='%1.2e')

def importNet(fileName):
  ind = np.loadtxt(fileName, delimiter=',')
  wMat = ind[:,:-1]     # Weight Matrix
  aVec = ind[:,-1]      # Activation functions

  # Create weight key
  wVec = wMat.flatten()
  wVec[np.isnan(wVec)]=0
  wKey = np.where(wVec!=0)[0] 

  return wVec, aVec, wKey


# Simple Wrapper for policy model
class NeatPolicy: 
    def __init__(self, indiv, game): 
        self.indiv = indiv 
        self.game = game 
        
        if self.indiv.aVec is None or self.indiv.aVec is not None and not isinstance(self.indiv.aVec, np.ndarray): 
            self.indiv.express()
        self.indiv.aVec[-1] = 1
        self.num_active_conn = self.indiv.conn[4,:].sum()
    
    def predict(self, input): 
        return act(self.indiv.wMat, self.indiv.aVec, self.game.input_size, self.game.output_size, input)[0]


class NEATPolicy: 
  def __init__(self, json_path): 
      from .ind import Ind 
      self.indiv = Ind.load(json_path)

      if self.indiv.aVec is None or self.indiv.aVec is not None and not isinstance(self.indiv.aVec, np.ndarray): 
          self.indiv.express()
      self.indiv.aVec[-1] = 1
    
      
  def predict(self, input): 
      return act(self.indiv.wMat, self.indiv.aVec, self.indiv.nInput, self.indiv.nOutput, input)[0]
      
      
def obtainOutgoingConnections(connG, node_id): 
  if connG is not None: 
    srcIndx = np.where(connG[1,:]==node_id)[0]
    exist = connG[2,srcIndx]
    exist = np.unique(exist).astype(int)
    return exist
  else:
    return []
  
  
  
# Sanity Checks 

def check_same_set(a, b): 
    if a.shape != b.shape: 
        return False 
    return set(a.tolist()) == set(b.tolist())
  
  
def sanity_check_node_func(nodeG, connG):
    
    wMat, node2order, node2seq, seq2node = getMat(nodeG, connG) # fixed an issue with node2order
    node = np.copy(nodeG)
    conn = np.copy(connG)
    nIns = len(node[0,node[1,:] == 1]) + len(node[0,node[1,:] == 4])
    nOuts = len(node[0,node[1,:] == 2])
    wMat, seq2node = calwMat(node, conn)
    order2seq = getMatOrder(nIns, nOuts, wMat)

    # wMat passes the check
    wMatSanityCheck(node,conn)
        
    # seq2order check
    print("seq2order Sanity Check: ")
    seq2order = {order2seq[seq_idx]: seq_idx for seq_idx in order2seq}
    for seq_idx in range(len(seq2order)):
        input_seq_ids = np.where((wMat[:, seq_idx] != 0) & (~np.isnan(wMat[:, seq_idx])))[0]
        for input_seq_id in input_seq_ids.tolist():
            if seq2order[input_seq_id] > seq2order[seq_idx]:
                print(f":: Error: Input Node {input_seq_id} (order {seq2order[input_seq_id]}) "
                    f"has higher order than Output Node {seq_idx} (order {seq2order[seq_idx]})")
                raise ValueError("Ordering sanity check failed for seq2order: ", seq_idx)
        print("  -- Ordering sanity check passed for seq_idx: ", seq_idx)
        
        
    # node2order check 
    print("node2order Sanity Check: ")
    for node_idx in range(len(node2order)):
        input_node_ids = conn[1, (conn[2,:]==node_idx) & (conn[4,:]==1)]
        for input_node_id in input_node_ids.tolist(): 
            if node2order[input_node_id] > node2order[node_idx]: 
                print(f"Error: Input Node {input_node_id} (order {node2order[input_node_id]}) "
                    f"has higher order than Output Node {node_idx} (order {node2order[node_idx]})")
                raise ValueError("Ordering sanity check failed for node2order: ", node_idx)
        print("  -- Ordering sanity check passed for node_idx: ", node_idx)
        
        
    print("Sanity Check Passed")
    
    
def wMatSanityCheck(node,conn):
    wMat, seq2node = calwMat(node, conn)

    connG, nodeG = np.copy(conn), np.copy(node)
    node2seq = [seq2node[seq_idx] for seq_idx in range(len(seq2node))]
    from fineNeat import check_same_set

    print("wMat Sanity Check: ")
    for node_idx in range(15): 
        seq_idx = node2seq[node_idx]
        to_ids_wMat = np.array([seq2node[int(i)] for i in np.where(wMat[seq_idx, :] != 0)[0]]) # inferred source node ids connected from node_idx | wMat 
        to_ids_connG = connG[2, connG[1,:]==node_idx]
        if not check_same_set(to_ids_wMat, to_ids_connG):
            raise ValueError("wMat Matching connG in target ids: ", check_same_set(to_ids_wMat, to_ids_connG))
        else: 
            print("  -- wMat Matching connG in target ids for node_idx: ", node_idx)

        from_ids_wMat = np.array([seq2node[int(i)] for i in np.where(wMat[:, seq_idx] != 0)[0]]) # inferred target node ids connected to node_idx | wMat 
        from_ids_connG = connG[1, connG[2,:]==node_idx]
        if not check_same_set(from_ids_wMat, from_ids_connG):
            raise ValueError("wMat Matching connG in source ids: ", check_same_set(from_ids_wMat, from_ids_connG))
        else: 
            print("  -- wMat Matching connG in source ids for node_idx: ", node_idx)
    print("wMat Sanity Check Passed")


def check_sparse_issue(node, conn):
    nodeG, connG = np.copy(node), np.copy(conn)
    input_node_ids = nodeG[0, nodeG[1,:] < 2]

    # Check connections FROM input/bias nodes that are disabled
    from_input_issues = connG[:, np.logical_and(np.isin(connG[1,:], input_node_ids), connG[4,:] == 0)]
    # Check connections TO input/bias nodes that are disabled
    to_input_issues = connG[:, np.logical_and(np.isin(connG[2,:], input_node_ids), connG[4,:] == 0)]

    issue_free = len(from_input_issues[0]) == 0 and len(to_input_issues[0]) == 0

    if not issue_free:
        print("Issues Found:")
        if len(from_input_issues[0]) > 0:
            print("\nDisabled connections FROM input/bias nodes:")
            for conn in from_input_issues.T:
                print(f"Connection {int(conn[0])}: From node {int(conn[1])} to node {int(conn[2])} is disabled")
        
        if len(to_input_issues[0]) > 0:
            print("\nDisabled connections TO input/bias nodes:")
            for conn in to_input_issues.T:
                print(f"Connection {int(conn[0])}: From node {int(conn[1])} to node {int(conn[2])} is disabled")
                
    return issue_free