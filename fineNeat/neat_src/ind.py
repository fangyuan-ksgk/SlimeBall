import numpy as np
import copy
from .ann import getLayer, getNodeOrder, obtainOutgoingConnections


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
    self.wMat    = []
    self.wVec    = []
    self.aVec    = []
    self.nConn   = []
    self.fitness = [] # Mean fitness over trials
    #self.fitMax  = [] # Best fitness over trials
    self.rank    = []
    self.birth   = []
    self.species = []

  def nConns(self):
    """Returns number of active connections
    """
    return int(np.sum(self.conn[4,:]))

  def express(self):
    """Converts genes to weight matrix and activation vector
    """
    order, wMat = getNodeOrder(self.node, self.conn)
    if order is not False: # no cyclic connections
      self.wMat = wMat
      self.aVec = self.node[2,order]

      wVec = self.wMat.flatten()
      wVec[np.isnan(wVec)] = 0
      self.wVec  = wVec
      self.nConn = np.sum(wVec!=0)
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

  def mutate(self,p,innov=None,gen=None):
    """Randomly alter topology and weights of individual

    Args:
      p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)
      child    - (Ind) - individual to be mutated
        .conns - (np_array) - connection genes
                 [5 X nUniqueGenes] 
                 [0,:] == Innovation Number (unique Id)
                 [1,:] == Source Node Id
                 [2,:] == Destination Node Id
                 [3,:] == Weight Value
                 [4,:] == Enabled?  
        .nodes - (np_array) - node genes
                 [3 X nUniqueGenes]
                 [0,:] == Node Id
                 [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                 [2,:] == Activation function (as int)
      innov    - (np_array) - innovation record
                 [5 X nUniqueGenes]
                 [0,:] == Innovation Number
                 [1,:] == Source
                 [2,:] == Destination
                 [3,:] == New Node?
                 [4,:] == Generation evolved

    Returns:
        child   - (Ind)      - newly created individual
        innov   - (np_array) - innovation record

    """
    # Readability
    nConn = np.shape(self.conn)[1]
    connG = np.copy(self.conn)
    nodeG = np.copy(self.node)

    
    # - Re-enable connections
    disabled  = np.where(connG[4,:] == 0)[0]
    reenabled = np.random.rand(1,len(disabled)) < p['prob_enable']
    connG[4,disabled] = reenabled
         
    # - Weight mutation
    # [Canonical NEAT: 10% of weights are fully random...but seriously?]
    mutatedWeights = np.random.rand(1,nConn) < p['prob_mutConn'] # Choose weights to mutate
    weightChange = mutatedWeights * np.random.randn(1,nConn) * p['ann_mutSigma'] # additive Gaussian noise  
    connG[3,:] += weightChange[0]
    
    # Clamp weight strength [ Warning given for nan comparisons ]  
    connG[3, (connG[3,:] >  p['ann_absWCap'])] =  p['ann_absWCap']
    connG[3, (connG[3,:] < -p['ann_absWCap'])] = -p['ann_absWCap']
    
    if (np.random.rand() < p['prob_addNode']) and np.any(connG[4,:]==1):
      connG, nodeG, innov = self.mutAddNode(connG, nodeG, innov, gen, p)
    
    if (np.random.rand() < p['prob_addConn']):
      connG, innov = self.mutAddConn(connG, nodeG, innov, gen, p) 
    
    child = Ind(connG, nodeG)
    child.birth = gen

    return child, innov

  def mutAddNode(self, connG, nodeG, innov, gen, p):
    """Add new node to genome

    Args:
      connG    - (np_array) - connection genes
                 [5 X nUniqueGenes] 
                 [0,:] == Innovation Number (unique Id) (Ascending order)
                 [1,:] == Source Node Id
                 [2,:] == Destination Node Id
                 [3,:] == Weight Value
                 [4,:] == Enabled?  
      nodeG    - (np_array) - node genes
                 [3 X nUniqueGenes]
                 [0,:] == Node Id
                 [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                 [2,:] == Activation function (as int)
      innov    - (np_array) - innovation record
                 [5 X nUniqueGenes]
                 [0,:] == Innovation Number (Ascending order)
                 [1,:] == Source
                 [2,:] == Destination
                 [3,:] == New Node?
                 [4,:] == Generation evolved
      gen      - (int) - current generation
      p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)


    Returns:
      connG    - (np_array) - updated connection genes
      nodeG    - (np_array) - updated node genes
      innov    - (np_array) - updated innovation record

    """
    # print(":: mutAddNode")
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
        
    # Disable original connection
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


    Todo: check for preexisting innovations to avoid duplicates in same gen

    Args:
      connG    - (np_array) - connection genes
                 [5 X nUniqueGenes] 
                 [0,:] == Innovation Number (unique Id)
                 [1,:] == Source Node Id
                 [2,:] == Destination Node Id
                 [3,:] == Weight Value
                 [4,:] == Enabled?  
      nodeG    - (np_array) - node genes
                 [3 X nUniqueGenes]
                 [0,:] == Node Id
                 [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                 [2,:] == Activation function (as int)
      innov    - (np_array) - innovation record
                 [5 X nUniqueGenes]
                 [0,:] == Innovation Number
                 [1,:] == Source
                 [2,:] == Destination
                 [3,:] == New Node?
                 [4,:] == Generation evolved
      gen      - (int)      - current generation
      p        - (dict)     - algorithm hyperparameters (see p/hypkey.txt)


    Returns:
      connG    - (np_array) - updated connection genes
      innov    - (np_array) - updated innovation record

    """
    # 1. never check existence within Innov 
    # 2. questionable use of setdiff1d ... 
    # print(":: mutAddConn")
    if innov is None:
      newConnId = connG[0,-1]+1
    else:
      newConnId = innov[0,-1]+1 

    nIns = len(nodeG[0,nodeG[1,:] == 1]) + len(nodeG[0,nodeG[1,:] == 4])
    nOuts = len(nodeG[0,nodeG[1,:] == 2])
    order, wMat = getNodeOrder(nodeG, connG)
    
    if order is False:
        # print(":: Failed to get node order")
        return connG, innov
        
    hMat = wMat[nIns:-nOuts,nIns:-nOuts]
    hLay = getLayer(hMat)+1
    
    if len(hLay) > 0:
        lastLayer = max(hLay)+1
    else:
        lastLayer = 1
        
    L = np.r_[np.zeros(nIns), hLay, np.full((nOuts),lastLayer)]
    
    nodeKey = np.c_[nodeG[0,order], L]

    sources = np.random.permutation(len(nodeKey)) # node index permutation
    for src in sources:
      srcLayer = nodeKey[src,1] # take source node according to index
      dest = np.where(nodeKey[:,1] > srcLayer)[0] # pick all nodes in higher layers
      
      # remove pre-existing outgoing connections
      exist_conn = obtainOutgoingConnections(connG, nodeKey[src, 0])
      exist_innov = obtainOutgoingConnections(innov, nodeKey[src, 0])
      exist = np.unique(np.hstack((exist_conn, exist_innov)))
      dest = np.setdiff1d(dest, exist)
      
      # Add a random valid connection
      np.random.shuffle(dest)
      if len(dest)>0:  # (there is a valid connection)
        # print(":: Successfully added connection")
        connNew = np.empty((5,1))
        connNew[0] = newConnId
        connNew[1] = nodeKey[src,0]
        connNew[2] = nodeKey[dest[0],0]
        connNew[3] = (np.random.rand()-0.5)*2*p['ann_absWCap']
        connNew[4] = 1
        connG = np.c_[connG,connNew]

        # Record innovation
        if innov is not None:
          newInnov = np.hstack((connNew[0:3].flatten(), -1, gen)) # (5,)
          innov = np.hstack((innov,newInnov[:,None])) # (5, ...)
        break;
      # else:
        # print(":: No valid connection found")

    return connG, innov

def initIndiv(shapes): 
  
    nodes = [shapes[0][0]] + [s[0] for s in shapes[1:]] + [shapes[-1][-1]]
    nInput = nodes[0]
    nOutput = nodes[-1]
    nHiddens = nodes[1:-1]
    nHidden = sum(nHiddens)
    nBias = nHidden + nOutput 
    nNode = nInput + nHidden + nOutput + nBias
        
    nodeId = np.arange(0, nNode)
    node = np.empty((3, nNode))
    node[0, :] = nodeId
    node[1, :nInput] = 1 
    node[1, nInput:nInput+nBias] = 4
    node[1, nInput:nInput+nBias+nHidden] = 3
    node[1, -nOutput:] = 2

    node[2, :] = 1

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
        id_in = np.where(raw_id_in >= nInput, raw_id_in + nInput, raw_id_in)
        id_out = np.where(raw_id_out >= nInput, raw_id_out + nInput, raw_id_out)
        
        conn[1, cum_conn: cum_conn + int(node_in * node_out)] = id_in
        conn[2, cum_conn: cum_conn + int(node_in * node_out)] = id_out
        
        cum_conn += int(node_in * node_out)
        cum_index += node_in
        
    cum_index = nWeight
    # Add Bias-Node Connection 
    for i, n_hidden in enumerate(nHiddens):
        conn[1, cum_index: cum_index + n_hidden] = np.arange(0, n_hidden) + cum_index - nWeight + nInput
        conn[2, cum_index: cum_index + n_hidden] = np.arange(0, n_hidden) + cum_index - nWeight + nInput + nBias
        cum_index += n_hidden
        
    # add bias to output nodes 
    conn[1, cum_index: cum_index + nOutput] = np.arange(0, nOutput) + cum_index - nWeight + nInput + nHidden
    conn[2, cum_index: cum_index + nOutput] = np.arange(0, nOutput) + cum_index - nWeight + nInput + nBias + nHidden 
            
    conn[0, :] = np.arange(0, nConn)
    conn[3, :] = np.random.randn(nConn) * 0.5
    conn[4, :] = 1
    
    return node, conn 
  
  
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