import numpy as np
import copy
import json
from .ann import getLayer, getNodeOrder, obtainOutgoingConnections, getNodeMap, getNodeInfo

def initIndiv(shapes): 
  
    nodes = [shapes[0][0]] + [s[0] for s in shapes[1:]] + [shapes[-1][-1]]
    nInput = nodes[0]
    nOutput = nodes[-1]
    nHiddens = nodes[1:-1]
    nHidden = sum(nHiddens)
    nBias = 1 
    nNode = nInput + nHidden + nOutput + nBias
        
    nodeId = np.arange(0, nNode)
    node = np.empty((3, nNode))
    node[0, :] = nodeId
    node[1, :nInput] = 1 
    node[1, nInput:nInput+nBias] = 4
    node[1, nInput+nBias:nInput+nBias+nHidden] = 3
    node[1, -nOutput:] = 2

    node[2, :] = 9 # relu activation pattern 

    nWeight = sum([s[0]*s[1] for s in shapes])
    nAddBias = nHidden + nOutput
    nConn = nWeight + nAddBias
    conn = np.empty((5, nConn))

    cum_conn = 0
    cum_index = 0

    # Add Node-Node Connection 
    for i, (node_in, node_out) in enumerate(shapes): 
        raw_id_in = np.tile(np.arange(0, node_in), node_out) + cum_index
        raw_id_out = np.repeat(np.arange(0, node_out), node_in) + cum_index + node_in
        
        # Convert raw IDs to actual node IDs
        id_in = np.where(raw_id_in >= nInput, raw_id_in + nBias, raw_id_in)
        id_out = np.where(raw_id_out >= nInput, raw_id_out + nBias, raw_id_out)
        
        conn_idx = np.arange(cum_conn, cum_conn + int(node_in * node_out))
        conn[1, conn_idx] = id_in
        conn[2, conn_idx] = id_out
        
        cum_conn += int(node_in * node_out)
        cum_index += node_in
        

    nWeight = cum_conn
    # Add Bias-Node Connection to hidden nodes
    for i, n_hidden in enumerate(nHiddens):
        id_in = nInput
        id_out = np.arange(0, n_hidden) + (cum_conn - nWeight) + nInput + nBias
        conn_idx = np.arange(cum_conn, cum_conn + n_hidden)
        conn[1, conn_idx] = id_in
        conn[2, conn_idx] = id_out
        cum_conn += n_hidden
        
    # add bias to output nodes 
    id_in = nInput
    id_out = np.arange(0, nOutput) + nInput + nBias + nHidden
    conn_idx = np.arange(cum_conn, cum_conn + nOutput)
    conn[1, conn_idx] = id_in
    conn[2, conn_idx] = id_out
        
    conn[0, :] = np.arange(0, nConn)
    conn[3, :] = np.random.randn(nConn) * 0.5
    conn[4, :] = 1
    
    return node, conn 
  
  
class Ind():
  """Individual class: genes, network, and fitness
  """ 
  def __init__(self, conn, node):
    """Intialize individual with given genes
    Args:
      conn - [5 X nUniqueGenes]
             [0,:] == Innovation Number
             [1,:] == Source
             [2,:] == Destination
             [3,:] == Weight
             [4,:] == Enabled?
      node - [3 X nUniqueGenes]
             [0,:] == Node Id
             [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
             [2,:] == Activation function (as int)
  
    Attributes:
      node    - (np_array) - node genes (see args)
      conn    - (np_array) - conn genes (see args)
      nInput  - (int)      - number of inputs
      nOutput - (int)      - number of outputs
      wMat    - (np_array) - weight matrix, one row and column for each node
                [N X N]    - rows: connection from; cols: connection to
      wVec    - (np_array) - wMat as a flattened vector
                [N**2 X 1]    
      aVec    - (np_array) - activation function of each node (as int)
                [N X 1]    
      nConn   - (int)      - number of connections
      fitness - (double)   - fitness averaged over all trials (higher better)
      X fitMax  - (double)   - best fitness over all trials (higher better)
      rank    - (int)      - rank in population (lower better)
      birth   - (int)      - generation born
      species - (int)      - ID of species
    """
    self.node    = np.copy(node)
    self.conn    = np.copy(conn)
    self.nInput  = sum(node[1,:]==1)
    self.nOutput = sum(node[1,:]==2)
    self.nBias = sum(node[1,:]==4)
    self.nHidden = sum(node[1,:]==3)
    
    self.wMat    = []
    self.wVec    = []
    self.aVec    = []
    self.nConn   = []
    self.fitness = [] # Mean fitness over trials
    #self.fitMax  = [] # Best fitness over trials
    self.rank    = []
    self.birth   = []
    self.species = []
    
    self.gen = 0
    
  @classmethod 
  def from_shapes(cls, shapes): 
    node, conn = initIndiv(shapes)
    return cls(conn, node)
  
  def to_params(self):  
    # Now run the parameter extraction code
    bias_idx = np.where(self.node[1,:] == 4)[0][0]
    node_map, orders, wMat = getNodeInfo(self.node, self.conn)
    layers = np.array([node_map[i][0] for i in range(len(node_map))])
    b_idx = node_map[bias_idx][1]

    params = []
    for layer_idx in range(max(layers)):
        curr_layer_nodes = (layers == layer_idx) & (np.arange(len(layers)) != bias_idx)
        next_layer_nodes = (layers == layer_idx + 1)
        
        curr_indices = np.array([node_map[i][1] for i, is_curr in enumerate(curr_layer_nodes) if is_curr])
        next_indices = np.array([node_map[i][1] for i, is_next in enumerate(next_layer_nodes) if is_next])
        
        layer_weight = wMat[curr_indices][:, next_indices]
        layer_bias = wMat[b_idx][next_indices]
        
        params.append((layer_weight, layer_bias))
        
    return params
  

  def nConns(self):
    """Returns number of active connections
    """
    return int(np.sum(self.conn[4,:]))

  def express(self, timeout=10):
    """
    Converts genes to nodeMap, order, and weight matrix | failed to express make current gene not expressable
    """
    node_map, order, wMat = getNodeInfo(self.node, self.conn, timeout=timeout) # cap on complexity here
        
    if order is not False: # no cyclic connections
      self.wMat = wMat
      self.aVec = self.node[2,order]

      wVec = self.wMat.flatten()
      wVec[np.isnan(wVec)] = 0
      self.wVec  = wVec
      self.nConn = np.sum(wVec!=0)
      
    if node_map is not False and order is not False: 
      self.max_layer = max([node_map[id][0] for id in node_map])
      self.node_map = node_map
      return True
    else:
      return False

  def createChild(self, p, innov, gen=0, mate=None):
    """Create new individual with this individual as a parent

      Args:
        p      - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
        innov  - (np_array) - innovation record
           [5 X nUniqueGenes]
           [0,:] == Innovation Number
           [1,:] == Source
           [2,:] == Destination
           [3,:] == New Node?
           [4,:] == Generation evolved
        gen    - (int)      - (optional) generation (for innovation recording)
        mate   - (Ind)      - (optional) second for individual for crossover


    Returns:
        child  - (Ind)      - newly created individual
        innov  - (np_array) - updated innovation record

    """  
    if mate is not None: # crossover if mate else mutation
      child = self.crossover(mate)
    else:
      child = Ind(self.conn, self.node)

    child, innov = child.mutate(p,innov,gen)
    return child, innov

# -- Canonical NEAT recombination operators ------------------------------ -- #

  def crossover(self,mate):
    """Combine genes of two individuals to produce new individual

      Procedure:
      ) Inherit all nodes and connections from most fit parent
      ) Identify matching connection genes in parentA and parentB
      ) Replace weights with parentB weights with some probability

      Args:
        parentA  - (Ind) - Fittest parent
          .conns - (np_array) - connection genes
                   [5 X nUniqueGenes]
                   [0,:] == Innovation Number (unique Id)
                   [1,:] == Source Node Id
                   [2,:] == Destination Node Id
                   [3,:] == Weight Value
                   [4,:] == Enabled?             
        parentB - (Ind) - Less fit parent

    Returns:
        child   - (Ind) - newly created individual

    """
    # Determine which parent is more fit
    if mate.fitness > self.fitness:
        parentA = mate    # Higher fitness parent
        parentB = self    # Lower fitness parent
    else:
        parentA = self    # Higher fitness parent
        parentB = mate    # Lower fitness parent

    # Inherit all nodes and connections from most fit parent
    child = Ind(parentA.conn, parentA.node)
    
    # Identify matching connection genes in ParentA and ParentB
    aConn = np.copy(parentA.conn[0,:])
    bConn = np.copy(parentB.conn[0,:])
    matching, IA, IB = np.intersect1d(aConn,bConn,return_indices=True)
    
    # Replace weights with parentB weights with some probability
    bProb = 0.5
    bGenes = np.random.rand(1,len(matching))<bProb
    child.conn[3,IA[bGenes[0]]] = parentB.conn[3,IB[bGenes[0]]]
    
    return child

  def safe_mutate(self, p):
    conn = self.conn 
    conn[3] += np.random.randn(conn[3].shape[0]) * 0.1
    node = self.node 
    child = Ind(conn, node) 
    assert child.express(), ":: Naive parameter mutation gives errored individual"
    return child
    
  def mutate(self,p,innov=None,gen=None, mute_top_change=False):
    """
    Randomly alter topology and weights of individual
    """
    # Readability
    nConn = np.shape(self.conn)[1]
    connG = np.copy(self.conn)
    nodeG = np.copy(self.node)

    
    # - Re-enable connections
    # if mute_top_change:
    #   disabled  = np.where(connG[4,:] == 0)[0]
    #   reenabled = np.random.rand(1,len(disabled)) < p['prob_enable']
    #   connG[4,disabled] = reenabled
         
    # - Weight mutation
    # [Canonical NEAT: 10% of weights are fully random...but seriously?]
    mutatedWeights = np.random.rand(1,nConn) < p['prob_mutConn'] # Choose weights to mutate
    weightChange = mutatedWeights * np.random.randn(1,nConn) * p['ann_mutSigma'] # additive Gaussian noise  
    connG[3,:] += weightChange[0]
    
    # Clamp weight strength [ Warning given for nan comparisons ]  
    connG[3, (connG[3,:] >  p['ann_absWCap'])] =  p['ann_absWCap']
    connG[3, (connG[3,:] < -p['ann_absWCap'])] = -p['ann_absWCap']
    
    gen = gen if gen is not None else self.gen if self.gen is not None else 0
    top_mutate = gen <= p['stop_topology_mutate_generations'] if 'stop_topology_mutate_generations' in p else 200
    top_mutate = top_mutate or mute_top_change
    
    if (np.random.rand() < p['prob_addNode'] * top_mutate) and np.any(connG[4,:]==1):
      connG, nodeG, innov = self.mutAddNode(connG, nodeG, innov, gen, p)
    
    if (np.random.rand() < p['prob_addConn'] * top_mutate):
      connG, nodeG, innov = self.mutAddConn(connG, nodeG, innov, gen, p) 
    
    child = Ind(connG, nodeG)
    child.birth = gen
    child.gen = gen + 1
    child_valid = child.express(timeout=p['timeout'] if 'timeout' in p else 50)
    if child_valid: 
      cap_layer = p['cap_layer'] if 'cap_layer' in p else 6
      child_valid = child_valid and (child.max_layer <= cap_layer)
    
    if not child_valid: 
      return False, False 
    else:
      return child, innov    

  def mutAddNode(self, connG, nodeG, innov, gen, p):
    """
    Add new node to genome
    """

    if innov is None:
      newNodeId = int(max(nodeG[0,:]+1))
      newConnId = connG[0,-1]+1    
    else:
      newNodeId = int(max(innov[2,:])+1) # next node id is a running counter
      newConnId = innov[0,-1]+1 
       
    # Choose connection to split
    connActive = np.where(connG[4,:] == 1)[0]
    if len(connActive) < 1:
      return connG, nodeG, innov # No active connections, nothing to split
    connSplit  = connActive[np.random.randint(len(connActive))]
    
    # Check if this split already exists in innovation record
    if innov is not None:
        srcNode = connG[1,connSplit]  # Source of connection being split
        dstNode = connG[2,connSplit]  # Destination of connection being split
        
        # Find all cases where a new node was added (innov[3,:] != -1)
        newNodeMask = innov[3,:] != -1
        # Among those, find where source and destination match
        matchingSrc = innov[1,newNodeMask] == srcNode
        matchingDst = innov[2,newNodeMask] == dstNode
        
        if np.any(matchingSrc & matchingDst):
            # This exact split already exists in innovation record
            # print(":: This exact split already exists in innovation record")
            return connG, nodeG, innov
          
    # Create new node
    newActivation = p['ann_actRange'][np.random.randint(len(p['ann_actRange']))]
    newNode = np.array([[newNodeId, 3, newActivation]]).T
    
    # Add connections to and from new node
    # -- Effort is taken to minimize disruption from node addition:
    # The 'weight to' the node is set to 1, the 'weight from' is set to the
    # original  weight. With a near linear activation function the change in performance should be minimal.

    
    
    connTo    = connG[:,connSplit].copy()
    connTo[0] = newConnId
    connTo[2] = newNodeId
    connTo[3] = 1 # weight set to 1
      
    connFrom    = connG[:,connSplit].copy()
    connFrom[0] = newConnId + 1
    connFrom[1] = newNodeId
    connFrom[3] = connG[3,connSplit] # weight set to previous weight value   
        
    newConns = np.vstack((connTo,connFrom)).T
        
    # Disable original connection :: aha I see, so it's still here but disabled
    connG[4,connSplit] = 0
        
    # Record innovations
    if innov is not None:
      newInnov = np.empty((5,2))
      newInnov[:,0] = np.hstack((connTo[0:3], newNodeId, gen))   
      newInnov[:,1] = np.hstack((connFrom[0:3], -1, gen)) 
      innov = np.hstack((innov,newInnov))
      
    # Add new structures to genome
    nodeG = np.hstack((nodeG,newNode))
    connG = np.hstack((connG,newConns))
    # print(":: Successfully added node")
    
    return connG, nodeG, innov

  def mutAddConn(self, connG, nodeG, innov, gen, p = {"ann_absWCap": 2}):
    """Add new connection to genome.
    To avoid creating recurrent connections all nodes are first sorted into
    layers, connections are then only created from nodes to nodes of the same or
    later layers.
    """

    if innov is None:
      newConnId = connG[0,-1]+1
    else:
      newConnId = innov[0,-1]+1 

    nodeMap = getNodeMap(nodeG, connG)
    if nodeMap is False:
        # print(":: Failed to get node order")
        return connG, nodeG, innov
      
    sources = np.random.permutation(list(nodeMap.keys()))
    for src_node_id in sources:
        src_node_layer = nodeMap[src_node_id][0] # take source node according to index
        dest_node_ids = [dest_node_id for dest_node_id in nodeMap if nodeMap[dest_node_id][0] > src_node_layer]
        
        # remove pre-existing outgoing connections
        exist_conn = obtainOutgoingConnections(connG, src_node_id)
        dest_node_ids = np.setdiff1d(dest_node_ids, exist_conn).astype(int)

        np.random.shuffle(dest_node_ids)
        if len(dest_node_ids)>0:  # (there is a valid connection)
            connNew = np.empty((5,1))
            connNew[0] = newConnId
            connNew[1] = src_node_id
            connNew[2] = dest_node_ids[0]
            connNew[3] = 1
            connNew[4] = 1
            connG = np.c_[connG,connNew]
                
            # Record innovation
            if innov is not None:
              newInnov = np.hstack((connNew[0:3].flatten(), -1, gen)) # (5,)
              innov = np.hstack((innov,newInnov[:,None])) # (5, ...)
            
            return connG, nodeG, innov
          
    return connG, nodeG, innov
        
  def save(self, filename): 
    with open(filename, 'w') as file: 
      json.dump({'conn': self.conn.tolist(), 'node': self.node.tolist()}, file)
    
  @classmethod 
  def load(cls, filename): 
    with open(filename, 'r') as file: 
      data = json.load(file)
      return cls(np.array(data['conn']), np.array(data['node']))
    
  @classmethod 
  def from_params(cls, params):
    """
    Still Buggy
    Initialize Ind class instance from params
    Risk losing 'innovation' trace
    """
    raise NotImplementedError
  
    # Count total nodes needed
    n_inputs = params[0][0].shape[0]  # First layer's input size
    n_outputs = params[-1][0].shape[1] # Last layer's output size
    n_hidden = sum(w.shape[1] for w, _ in params[:-1])  # Hidden nodes across layers
    n_bias = 1
    n_total = n_inputs + n_hidden + n_outputs + n_bias

    # Create node genes [id, type, activation]
    node = np.empty((3, n_total))
    node[0, :] = np.arange(n_total)  # Node IDs
    
    # Set node types
    node[1, :n_inputs] = 1  # Input nodes
    node[1, n_inputs:n_inputs+n_bias] = 4  # Bias node
    node[1, n_inputs+n_bias:n_inputs+n_bias+n_hidden] = 3  # Hidden nodes
    node[1, -n_outputs:] = 2  # Output nodes
    
    # Set all activations to 1 (can be modified if needed)
    node[2, :] = 1

    # Count total connections needed
    n_weights = sum(w.size for w, _ in params)  # Regular weights
    n_bias_conns = n_hidden + n_outputs  # Bias connections
    n_total_conns = n_weights + n_bias_conns

    # Create connection genes [innov, source, dest, weight, enabled]
    conn = np.empty((5, n_total_conns))
    conn[0, :] = np.arange(n_total_conns)  # Innovation numbers
    conn[4, :] = 1  # All connections enabled

    # Fill in connections layer by layer
    curr_conn = 0
    curr_node = n_inputs + n_bias  # Start after inputs and bias

    # Add regular connections
    for w, _ in params:
        rows, cols = w.shape
        
        # Get source and destination node indices
        if curr_node == n_inputs + n_bias:  # First layer
            sources = np.arange(n_inputs)
        else:
            sources = np.arange(curr_node - cols, curr_node)
            
        dests = np.arange(curr_node, curr_node + cols)
        
        # Create all connections between layers
        src_idx = np.repeat(sources, cols)
        dest_idx = np.tile(dests, rows)
        weights = w.flatten()
        
        n_conns = len(src_idx)
        conn_slice = slice(curr_conn, curr_conn + n_conns)
        conn[1, conn_slice] = src_idx
        conn[2, conn_slice] = dest_idx
        conn[3, conn_slice] = weights
        
        curr_conn += n_conns
        curr_node += cols

    # Add bias connections
    bias_node = n_inputs
    hidden_and_output_nodes = np.arange(n_inputs + n_bias, n_total)
    conn[1, -n_bias_conns:] = bias_node
    conn[2, -n_bias_conns:] = hidden_and_output_nodes
    conn[3, -n_bias_conns:] = np.random.randn(n_bias_conns) * 0.5

    return cls(conn, node)




  
  
def calculate_layers(graph):
    nodes = set(graph['node'])
    edges = graph['edge']
    
    # Create adjacency dictionary for predecessors
    predecessors = {node: [] for node in nodes}
    for start, end in edges:
        predecessors[end].append(start)
    
    # Initialize layers
    layers = {}
    
    # Find input nodes (nodes with no predecessors)
    input_nodes = {node for node in nodes if not predecessors[node]}
    for node in input_nodes:
        layers[node] = 0
        
    # Process remaining nodes
    while len(layers) < len(nodes):
        for node in nodes:
            if node not in layers:
                # Check if all predecessors have layers assigned
                if all(pred in layers for pred in predecessors[node]):
                    # Layer is max of predecessors' layers plus 1
                    layers[node] = max((layers[pred] for pred in predecessors[node]), default=-1) + 1
    
    return layers