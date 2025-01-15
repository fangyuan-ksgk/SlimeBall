import numpy as np
import torch
import torch.nn as nn


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
  conn = np.copy(connG)
  node = np.copy(nodeG)
  nIns = len(node[0,node[1,:] == 1]) + len(node[0,node[1,:] == 4])
  nOuts = len(node[0,node[1,:] == 2])

  # Create connection and initial weight matrices
  conn[3,conn[4,:]==0] = np.nan # disabled but still connected
  src  = conn[1,:].astype(int)
  dest = conn[2,:].astype(int)

  # Reordering node : input, bias, output, hidden ?? should be input, bias, hidden, output ??
  reordered_index = np.r_[node[0, node[1, :] ==1], node[0,node[1,:] == 4], node[0,node[1,:] == 3], node[0,node[1,:] == 2]]

  # Get Edge on reordered nodes 
  src_mask = (src.reshape(-1, 1) == reordered_index.reshape(1, -1)) # (n_conn, n_node)
  dest_mask = (dest.reshape(-1, 1) == reordered_index.reshape(1, -1))
  src = (src_mask @ np.arange(len(reordered_index)).reshape(-1, 1)).flatten()  # Convert to 1D
  dest = (dest_mask @ np.arange(len(reordered_index)).reshape(-1, 1)).flatten()  # Convert to 1D

  # Create weight matrix according to reordered nodes
  wMat = np.zeros((np.shape(node)[1],np.shape(node)[1]))
  wMat[src,dest] = conn[3,:] # assign weight to the connection

  # Get connection matrix (connection between hidden nodes)
  connMat = wMat[nIns+nOuts:,nIns+nOuts:]
  connMat[connMat!=0] = 1

  # Topological Sort of Hidden Nodes (according to connection matrix)
  # Q : sorted "local index" of hidden nodes (smallest index 0)
  edge_in = np.sum(connMat,axis=0) # sum of edges ending with each node
  Q = np.where(edge_in==0)[0]  # array of node ids with no incoming connections
  for i in range(len(connMat)):
      if (len(Q) == 0) or (i >= len(Q)):
          Q = []
          return False, False # Cycle found, can't sort
        
      edge_out = connMat[Q[i],:]
      edge_in  = edge_in - edge_out # Remove previous layer nodes' conns from total
      nextNodes = np.setdiff1d(np.where(edge_in==0)[0], Q) # Exclude previous layer nodes
      Q = np.hstack((Q,nextNodes)) # Add next layer nodes to Q

      if sum(edge_in) == 0:
          break

  # Add In and outs back and reorder wMat according to sort
  Q += nIns+nOuts # Shifted local index due to reordering (input, bias, output, hidden) 

  Q = np.r_[np.arange(nIns),              
          Q,                              
          np.arange(nIns,nIns+nOuts)]     
  
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
      lastLayer = max(hLay)+1
  else:
      lastLayer = 1
      
  L = np.r_[np.zeros(nIns), hLay, np.full((nOuts),lastLayer)]
  nodeKey = np.c_[nodeG[0,order], L]
  
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
      
  L = np.r_[np.zeros(nIns), hLay, np.full((nOuts),lastLayer)].astype(int)
  
  nodeMap = {}
  for i in range(len(nodeG[0])): 
      nodeMap[order[i]] = [L[i], i] # layer index, order index
    
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
      
  L = np.r_[np.zeros(nIns), hLay, np.full((nOuts),lastLayer)].astype(int)
  
  nodeMap = {}
  for i in range(len(nodeG[0])): 
      nodeMap[order[i]] = [L[i], i] # layer index, order index
    
  return nodeMap, order, wMat


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
  
  
