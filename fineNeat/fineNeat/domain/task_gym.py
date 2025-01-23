import random
import numpy as np
import sys
from .make_env import make_env
from ..neat_src import act, selectAct
from fineNeat.neat_src.ann import NeatPolicy
from slimevolleygym import multiagent_rollout as rollout


# General Wrapper class for Gym environments

class GymTask():
  """Problem domain to be solved by neural network. Uses OpenAI Gym patterns.
  """ 
  def __init__(self, game, paramOnly=False, nReps=1): 
    """Initializes task environment
  
    Args:
      game - (string) - dict key of task to be solved (see domain/config.py)
  
    Optional:
      paramOnly - (bool)  - only load parameters instead of launching task?
      nReps     - (nReps) - number of trials to get average fitness
    """
    # Network properties
    self.nInput   = game.input_size
    self.nOutput  = game.output_size      
    self.actRange = game.h_act
    self.absWCap  = game.weightCap
    self.layers   = game.layers      
    self.activations = np.r_[np.full(1,1),game.i_act,game.o_act]
  
    # Environment
    self.nReps = nReps
    self.maxEpisodeLength = game.max_episode_length
    self.actSelect = game.actionSelect
    if not paramOnly:
      self.env = make_env(game.env_name)
    
    # Special needs...
    self.needsClosed = (game.env_name.startswith("CartPoleSwingUp"))  
    
    # Special needs (II). 
    self.slime = game.env_name.startswith("SlimeVolley")
    # if self.slime:
    #   from slimevolleygym import BaselinePolicy
    #   self.opponent = BaselinePolicy()
  
  def getFitness(self, wVec, aVec, hyp=None, view=False, nRep=False, seed=-1):
    """Get fitness of a single individual.
  
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
  
    Optional:
      view    - (bool)     - view trial?
      nReps   - (nReps)    - number of trials to get average fitness
      seed    - (int)      - starting random seed for trials
  
    Returns:
      fitness - (float)    - mean reward over all trials
    """
    if nRep is False:
      nRep = self.nReps
    wVec[np.isnan(wVec)] = 0
    reward = np.empty(nRep)
    for iRep in range(nRep):
      reward[iRep] = self.testInd(wVec, aVec, view=view, seed=seed+iRep)
    fitness = np.mean(reward)
    return fitness
  
  def getTournamentScore(self, ind_left, ind_right_list): 
    """
    Get tournament score for an individual against a list of individuals
    """
    scores = []
    for ind_right in ind_right_list: 
      score_left, score_right = self.match_score(ind_left, ind_right)
      scores.append(score_left)
    fitness = np.mean(scores)
    return fitness
    
  def testInd(self, wVec, aVec, view=False,seed=-1):
    """Evaluate individual on task
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
  
    Optional:
      view    - (bool)     - view trial?
      seed    - (int)      - starting random seed for trials
  
    Returns:
      fitness - (float)    - reward earned in trial
    """
    if seed >= 0:
      random.seed(seed)
      np.random.seed(seed)
      self.env.seed(seed)
    state = self.env.reset()
    self.env.t = 0
    annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
    action = selectAct(annOut, self.actSelect)          
    if self.slime:
        action = [1 if x > 0 else 0 for x in annOut[0]]  # Convert each output to binary
   
    wVec[wVec!=0]
    predName = str(np.mean(wVec[wVec!=0]))      
    state, reward, done, info = self.env.step(action)
    
    if self.maxEpisodeLength == 0:
      if view:
        if self.needsClosed:
          self.env.render(close=done)  
        else:
          self.env.render()
      return reward
    else:
      totalReward = reward
    
    for tStep in range(self.maxEpisodeLength): 
      annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
    
      action = selectAct(annOut, self.actSelect) 
      if self.slime:
        action = [1 if x > 0 else 0 for x in annOut[0]]  # Convert each output to binary

      state, reward, done, info = self.env.step(action)
      totalReward += reward  
      if view:
        if self.needsClosed:
          self.env.render(close=done)  
        else:
          self.env.render()
      if done:
        break
      
    totalReward = totalReward * (1 - tStep / self.maxEpisodeLength) # discount reward with time
    return totalReward
  
  def match_score(self, ind_left, ind_right): 
    assert self.slime, "SlimeVolley is required for match_score"
    from fineNeat.domain.config import games
    game = games["slimevolley"]
    policy_left = NeatPolicy(ind_left, game)
    policy_right = NeatPolicy(ind_right, game)
    score_right, t = rollout(self.env, policy_left, policy_right)
    score_left = -score_right
    # reward faster winning or slower losing
    score_left = add_dual_agent_time_score(score_left, t, max_steps = self.env.t_limit)
    score_right = add_dual_agent_time_score(score_right, t, max_steps = self.env.t_limit)
    return score_left, score_right


def add_dual_agent_time_score(score, t, max_steps): 
  """ 
  For dual-agent game, reward faster winning or slower losing
  """
  time_factor = 0.5 * (max_steps - t) / max_steps  # Scale factor based on time taken
  if score > 0: 
    return score + time_factor
  else: 
    return score - time_factor