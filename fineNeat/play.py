import warnings
# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import os
import numpy as np
import pickle
import argparse
import slimevolleygym
from fineNeat.neat_src.ann import NEATPolicy
from slimevolleygym.mlp import makeSlimePolicy, makeSlimePolicyLite # simple pretrained models
from slimevolleygym import BaselinePolicy
from time import sleep
# import neat

#import cv2

np.set_printoptions(threshold=20, precision=4, suppress=True, linewidth=200)

PPO1 = None # from stable_baselines import PPO1 (only load if needed.)
class PPOPolicy:
  def __init__(self, path):
    self.model = PPO1.load(path)
  def predict(self, obs):
    action, state = self.model.predict(obs, deterministic=True)
    return action

class RandomPolicy:
  def __init__(self, path):
    self.action_space = gym.spaces.MultiBinary(3)
    pass
  def predict(self, obs):
    return self.action_space.sample()

def makeBaselinePolicy(_):
  return BaselinePolicy()


def rollout(env, policy0, policy1, render_mode=False):
  """ play one agent vs the other in modified gym-style loop. """
  obs0 = env.reset()
  obs1 = obs0 # same observation at the very beginning for the other agent

  done = False
  total_reward = 0
  #count = 0

  while not done:

    # agent absolute x,y,vx,xy | ball position x,y,vx,vy | opponent absolute x,y,vx,vy
    # Therefore, compared to agent1 observation, agent0 observation flipps last 4 with first 4 and negates ball's x & vx 
    action0 = policy0.predict(obs0)
    action1 = policy1.predict(obs1)

    # uses a 2nd (optional) parameter for step to put in the other action
    # and returns the other observation in the 4th optional "info" param in gym's step()
    obs0, reward, done, info = env.step(action0, action1)
    obs1 = info['otherObs']

    total_reward += reward

    if render_mode:
      env.render()
      """ # used to render stuff to a gif later.
      img = env.render("rgb_array")
      filename = os.path.join("gif","daytime",str(count).zfill(8)+".png")
      cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
      count += 1
      """
      sleep(0.01)

  return total_reward

def evaluate_multiagent(env, policy0, policy1, render_mode=False, n_trials=1000, init_seed=721):
  history = []
  for i in range(n_trials):
    env.seed(seed=init_seed+i)
    cumulative_score = rollout(env, policy0, policy1, render_mode=render_mode)
    print("cumulative score #", i, ":", cumulative_score)
    history.append(cumulative_score)
  return history


import glob 
  
def get_neat_file(file_path: str) -> str: 
  if file_path.endswith(".json"): 
    return file_path
  else: 
    files = glob.glob(file_path + "/*.json")  
    if len(files) == 0:
      return None
    if "volley" in file_path: 
      files.sort(key=lambda x: int(x.split("volley")[-1].split("_best")[0]))
    elif "sneat" in file_path:
      files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return files[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_logdir', type=str, required=True, help='Directory containing NEAT checkpoint files')
    parser.add_argument('--right_logdir', type=str, required=True, help='Directory containing NEAT checkpoint files')
    args = parser.parse_args()

    env = gym.make("SlimeVolley-v0")
    import time

    while True:
        # Get latest checkpoint file
        left_neat_file = get_neat_file(args.left_logdir)
        right_neat_file = get_neat_file(args.right_logdir)

        if left_neat_file is None or right_neat_file is None:
            time.sleep(100)
            continue

        policy0 = NEATPolicy(right_neat_file)
        policy1 = NEATPolicy(left_neat_file)
        
        # Evaluate the agents against each other
        history = evaluate_multiagent(env, policy0, policy1, render_mode=True, n_trials=5)
        print(f"Left Agent checkpoint: {left_neat_file}", "   |   ", f"Right Agent checkpoint: {right_neat_file}")