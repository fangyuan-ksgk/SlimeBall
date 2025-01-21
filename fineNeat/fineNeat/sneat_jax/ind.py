import jax.numpy as jnp
import copy
import json
from .ann import obtainOutgoingConnections, getNodeInfo
import jax

def initIndiv(shapes, seed=0): 
  
    nodes = [shapes[0][0]] + [s[0] for s in shapes[1:]] + [shapes[-1][-1]]
    nInput = nodes[0]
    nOutput = nodes[-1]
    nHiddens = nodes[1:-1]
    nHidden = sum(nHiddens)
    nBias = 1 
    nNode = nInput + nHidden + nOutput + nBias
        
    nodeId = jnp.arange(0, nNode)
    node = jnp.empty((3, nNode))
    node = node.at[0, :].set(nodeId)
    node = node.at[1, :nInput].set(1)
    node = node.at[1, nInput:nInput+nBias].set(4)
    node = node.at[1, nInput+nBias:nInput+nBias+nHidden].set(3)
    node = node.at[1, -nOutput:].set(2)

    node = node.at[2, :].set(9)  # relu activation pattern 

    nWeight = sum([s[0]*s[1] for s in shapes])
    nAddBias = nHidden + nOutput
    nConn = nWeight + nAddBias
    conn = jnp.empty((5, nConn))

    cum_conn = 0
    cum_index = 0

    # Add Node-Node Connection 
    for i, (node_in, node_out) in enumerate(shapes): 
        raw_id_in = jnp.tile(jnp.arange(0, node_in), node_out) + cum_index
        raw_id_out = jnp.repeat(jnp.arange(0, node_out), node_in) + cum_index + node_in
        
        # Convert raw IDs to actual node IDs
        id_in = jnp.where(raw_id_in >= nInput, raw_id_in + nBias, raw_id_in)
        id_out = jnp.where(raw_id_out >= nInput, raw_id_out + nBias, raw_id_out)
        
        conn_idx = jnp.arange(cum_conn, cum_conn + int(node_in * node_out))
        conn = conn.at[1, conn_idx].set(id_in)
        conn = conn.at[2, conn_idx].set(id_out)
        
        cum_conn += int(node_in * node_out)
        cum_index += node_in

    nWeight = cum_conn
    # Add Bias-Node Connection to hidden nodes
    for i, n_hidden in enumerate(nHiddens):
        id_in = nInput
        id_out = jnp.arange(0, n_hidden) + (cum_conn - nWeight) + nInput + nBias
        conn_idx = jnp.arange(cum_conn, cum_conn + n_hidden)
        conn = conn.at[1, conn_idx].set(id_in)
        conn = conn.at[2, conn_idx].set(id_out)
        cum_conn += n_hidden
        
    # add bias to output nodes 
    id_in = nInput
    id_out = jnp.arange(0, nOutput) + nInput + nBias + nHidden
    conn_idx = jnp.arange(cum_conn, cum_conn + nOutput)
    conn = conn.at[1, conn_idx].set(id_in)
    conn = conn.at[2, conn_idx].set(id_out)
        
    conn = conn.at[0, :].set(jnp.arange(0, nConn))
    key = jax.random.PRNGKey(seed)  # You'll need to pass this key as a parameter
    conn = conn.at[3, :].set(jax.random.normal(key, (nConn,)) * 0.5)
    conn = conn.at[4, :].set(1)
    
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
    self.node    = jnp.copy(node)
    self.conn    = jnp.copy(conn)
    self.nInput  = sum(node[1,:]==1).item()
    self.nOutput = sum(node[1,:]==2).item()
    self.nBias = sum(node[1,:]==4).item()
    self.nHidden = sum(node[1,:]==3).item()
    
    self.wMat    = []
    self.wVec    = []
    self.aVec    = []
    self.nConn   = []
    self.fitness = -jnp.inf
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
    bias_idx = jnp.where(self.node[1,:] == 4)[0][0].item()
    node_map, orders, wMat = getNodeInfo(self.node, self.conn)
    layers = jnp.array([node_map[i][0] for i in range(len(node_map))])
    b_idx = node_map[bias_idx][1]

    params = []
    for layer_idx in range(max(layers)):
        curr_layer_nodes = (layers == layer_idx) & (jnp.arange(len(layers)) != bias_idx)
        next_layer_nodes = (layers == layer_idx + 1)
        
        curr_indices = jnp.array([node_map[i][1] for i, is_curr in enumerate(curr_layer_nodes) if is_curr])
        next_indices = jnp.array([node_map[i][1] for i, is_next in enumerate(next_layer_nodes) if is_next])
        
        layer_weight = wMat[curr_indices][:, next_indices]
        layer_bias = wMat[b_idx][next_indices]
        
        params.append((layer_weight, layer_bias))
        
    return params

  def nConns(self):
    """Returns number of active connections
    """
    return int(jnp.sum(self.conn[4,:]))

  def express(self, timeout=10):
    """
    Converts genes to nodeMap, order, and weight matrix | failed to express make current gene not expressable
    """
    node_map, order, wMat = getNodeInfo(self.node, self.conn) # cap on complexity here
        
    if order is not False: # no cyclic connections
      self.wMat = wMat
      self.aVec = self.node[2,order]

      wVec = self.wMat.flatten()
      wVec = jnp.nan_to_num(wVec, 0.0)
      self.wVec  = wVec
      self.nConn = jnp.sum(wVec!=0)
      
    if node_map is not False and order is not False: 
      self.max_layer = max([node_map[id][0] for id in node_map])
      self.node_map = node_map
      return True
    else:
      return False
  
  def mutate(self, p, innov=None, gen=None, mute_top_change=True, seed=0):
    """
    Randomly alter topology and weights of individual
    """
    # Readability
    nConn = jnp.shape(self.conn)[1]
    connG = jnp.copy(self.conn)
    nodeG = jnp.copy(self.node)
    
    innov_orig = jnp.copy(innov) if innov is not None else None
    
    # - Change connection status (Turn On/Off)
    connG, nodeG, innov = self.mutSparsity(p, innov, seed=seed)
         
    # - Weight mutation
    key = jax.random.PRNGKey(seed)
    mutatedWeights = jax.random.uniform(key, shape=(1, nConn)) < p['prob_mutConn'] # Choose weights to mutate
    weightChange = mutatedWeights * jax.random.normal(key, shape=(1, nConn)) * p['ann_mutSigma'] # additive Gaussian noise  
    connG = connG.at[3, :].set(connG[3] + weightChange[0])
    
    # - Clamp weight strength
    connG = connG.at[3, (connG[3,:] >  p['ann_absWCap'])].set(p['ann_absWCap'])
    connG = connG.at[3, (connG[3,:] < -p['ann_absWCap'])].set(-p['ann_absWCap'])
    
    # Cap on number of layers & connections
    active_conn = jnp.sum(connG[4,:])
    top_mutate = active_conn < p['cap_conn']
    
    key = jax.random.PRNGKey(seed + 1)
    if (jax.random.uniform(key, shape=()) < p['prob_addNode'] * top_mutate) and jnp.any(connG[4,:]==1):
      connG, nodeG, innov = self.mutAddNode(connG, nodeG, innov, gen, p, seed=seed + 2)
    
    key = jax.random.PRNGKey(seed + 3)
    if (jax.random.uniform(key, shape=()) < p['prob_addConn'] * top_mutate):
      connG, nodeG, innov = self.mutAddConn(connG, nodeG, innov, gen, p, seed=seed + 4) 
    
    child = Ind(connG, nodeG)
    child.birth = gen
    child.gen = gen + 1 if gen is not None else None
    
    child_valid = child.express(timeout=p['timeout'] if 'timeout' in p else 10)
    
    if child_valid: 
      return child, innov 
    else:
      print(":: Failed to express child")
      return self, innov_orig 
  
  def safe_mutate(self, seed=0):
    conn = self.conn 
    key = jax.random.PRNGKey(seed)  # Added key for random number generation
    conn = conn.at[3, :].set(conn[3] + jax.random.normal(key, shape=(conn[3].shape)) * 0.1)
    node = self.node 
    child = Ind(conn, node) 
    assert child.express(), ":: Naive parameter mutation gives errored individual"
    return child
  
  def mutAddNode(self, connG, nodeG, innov, gen, p, seed=0):
    """
    Add new node to genome
    """

    if innov is None:
      newNodeId = int(max(nodeG[0,:]+1))
      newConnId = connG[0,-1]+1    
    else:
      newNodeId = int(max(innov[2,:])+1) # next node id is a running counter
      newConnId = innov[0,-1]+1 
       
    connActive = jnp.where(connG[4,:] == 1)[0]
    if len(connActive) < 1:
      return connG, nodeG, innov # No active connections, nothing to split
  
    key = jax.random.PRNGKey(seed)
    connSplit = connActive[jax.random.randint(key, shape=(), minval=0, maxval=len(connActive))]

    if innov is not None:
        srcNode = connG[1,connSplit]  # Source of connection being split
        dstNode = connG[2,connSplit]  # Destination of connection being split
        
        newNodeMask = innov[3,:] != -1
        matchingSrc = innov[1,newNodeMask] == srcNode
        matchingDst = innov[2,newNodeMask] == dstNode
        
        if jnp.any(matchingSrc & matchingDst):
            return connG, nodeG, innov
          
    # Create new node
    key = jax.random.PRNGKey(seed + 1)
    newActivation = p['ann_actRange'][jax.random.randint(key, shape=(), minval=0, maxval=len(p['ann_actRange']))]
    newNode = jnp.array([[newNodeId, 3, newActivation]]).T
    
    connTo    = connG[:,connSplit].copy()
    connTo = connTo.at[0].set(newConnId)
    connTo = connTo.at[2].set(newNodeId)
    connTo = connTo.at[3].set(1) # weight set to 1
      
    connFrom    = connG[:,connSplit].copy()
    connFrom = connFrom.at[0].set(newConnId + 1)
    connFrom = connFrom.at[1].set(newNodeId)
    connFrom = connFrom.at[3].set(connG[3,connSplit]) # weight set to previous weight value   
        
    newConns = jnp.vstack((connTo,connFrom)).T
        
    # Set original connection weight to 0.0 instead of disabling it
    connG = connG.at[3,connSplit].set(0.0)
        
    # Record innovations
    if innov is not None:
      newInnov = jnp.empty((5,2))
      newInnov = newInnov.at[0].set(jnp.hstack((connTo[0:3], newNodeId, gen)))   
      newInnov = newInnov.at[1].set(jnp.hstack((connFrom[0:3], -1, gen))) 
      innov = jnp.hstack((innov,newInnov))
      
    # Add new structures to genome
    nodeG = jnp.hstack((nodeG,newNode))
    connG = jnp.hstack((connG,newConns))
    
    return connG, nodeG, innov

  def mutSparsity(self, p, innov=None, seed=0):
    nodeG = jnp.copy(self.node)
    connG = jnp.copy(self.conn)
    nodeMap, _, _ = getNodeInfo(nodeG, connG)
    if nodeMap is False:
        print(":: Failed to get node order")
        return connG, nodeG, innov

    # pick non-essential connections and pick ratio of them to randomize 'on/off' status 
    bias_node_ids = nodeG[0, (nodeG[1,:]==4) | (nodeG[1,:]==1)]
    non_essential_conn_ids = ~jnp.isin(connG[1,:], bias_node_ids) & ~jnp.isin(connG[2,:], bias_node_ids)

    # Randomly select connections to modify based on change_ratio
    n_conns = jnp.sum(non_essential_conn_ids)
    n_change = int(n_conns * p['prob_mutTurnConn'])
    
    key = jax.random.PRNGKey(seed)
    change_indices = jax.random.choice(key, n_conns, shape=(n_change,), replace=False)
    
    # Create array of 1s and 0s based on sparsity ratio
    key = jax.random.PRNGKey(seed + 1)
    new_states = jax.random.bernoulli(key, p=p['sparsity_ratio'], shape=(n_change,))

    # Update selected connections
    update_indices = jnp.arange(connG.shape[1])[non_essential_conn_ids][change_indices]
    connG = connG.at[4, update_indices].set(new_states)
    return connG, nodeG, innov
  
  
  def mutAddConn(self, connG, nodeG, innov, gen, p = {"ann_absWCap": 2}, seed=0):
    """Add new connection to genome.
    To avoid creating recurrent connections all nodes are first sorted into
    layers, connections are then only created from nodes to nodes of the same or
    later layers.
    """

    if innov is None:
      newConnId = connG[0,-1]+1
    else:
      newConnId = innov[0,-1]+1 

    nodeMap, _, _ = getNodeInfo(nodeG, connG)
    if nodeMap is False:
        # print(":: Failed to get node order")
        return connG, nodeG, innov
      
    key = jax.random.PRNGKey(seed)
    sources = jax.random.permutation(key, jnp.array(list(nodeMap.keys())))
    for src_node_id in sources:
        src_node_id = int(src_node_id)
        src_node_layer = nodeMap[src_node_id][0] # take source node according to index
        dest_node_ids = [dest_node_id for dest_node_id in nodeMap if nodeMap[dest_node_id][0] > src_node_layer]
        
        # remove pre-existing outgoing connections
        exist_conn = obtainOutgoingConnections(connG, src_node_id)
        # Convert list to JAX array before using setdiff1d
        dest_node_ids = jnp.array(dest_node_ids)
        dest_node_ids = jnp.setdiff1d(dest_node_ids, exist_conn).astype(int)

        key = jax.random.PRNGKey(seed + 1)
        dest_node_ids = jax.random.permutation(key, dest_node_ids, independent=True)
        if len(dest_node_ids)>0:  # (there is a valid connection)
            connNew = jnp.empty((5,1))
            connNew = connNew.at[0].set(newConnId)
            connNew = connNew.at[1].set(src_node_id)
            connNew = connNew.at[2].set(dest_node_ids[0])
            connNew = connNew.at[3].set(1)
            connNew = connNew.at[4].set(1)
            connG = jnp.c_[connG,connNew]
                
            # Record innovation
            if innov is not None:
              newInnov = jnp.hstack((connNew[0:3].flatten(), -1, gen)) # (5,)
              innov = jnp.hstack((innov,newInnov[:,None])) # (5, ...)
            
            return connG, nodeG, innov
          
    return connG, nodeG, innov
  
  def save(self, filename): 
    with open(filename, 'w') as file: 
      json.dump({'conn': self.conn.tolist(), 'node': self.node.tolist()}, file)
    
  @classmethod 
  def load(cls, filename): 
    with open(filename, 'r') as file: 
      data = json.load(file)
      return cls(jnp.array(data['conn']), jnp.array(data['node']))
    
  def to_np(self): 
    from ..neat_src.ind import Ind
    import numpy as np 
    ind = Ind(np.copy(self.conn), np.copy(self.node))
    ind.express()
    return ind 