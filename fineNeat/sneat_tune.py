# Evolve from best sneat agent to beat baseline
# - through evaluating against miscellaneous opponents
# - as an experiment, I'd like to see if evolve sneat agent to beat baseline works ... 
# - essentially we'd keep mutating it until it beats baseline 

import os
import numpy as np
import gym
from slimevolleygym.mlp import games as games
from slimevolleygym.mlp import Model
from slimevolleygym import multiagent_rollout as rollout
from fineNeat import loadHyp, load_task, updateHyp, Ind, NeatPolicy, viewInd, fig2img, NEATPolicy
import matplotlib.pyplot as plt


# Settings
random_seed = 612
population_size = 256
total_tournaments = 150000
save_freq = 1000

# Environment

# Hyperparameters
hyp_default = 'fineNeat/p/default_sneat.json'
hyp_adjust = "fineNeat/p/volley.json"

hyp = loadHyp(pFileName=hyp_default, load_task=load_task)
updateHyp(hyp,load_task,hyp_adjust)

# Log results
logdir = "../runs/sneat_tune"
visdir = "../runs/sneat_tune/vis"
if not os.path.exists(logdir):
  os.makedirs(logdir)
if not os.path.exists(visdir):
  os.makedirs(visdir)
  

def mutate(ind, p): 
    child, _ = ind.mutate(p=p)
    if child: 
       return child 
    else:
        return ind.safe_mutate(p)


game = games['slimevolleylite']
# load best sneat agent into population 
best_sneat_json = "../runs/sneat_sp/sneat_00951000.json"
best_sneat_ind = Ind.load(best_sneat_json)
population = [best_sneat_ind.safe_mutate(hyp) for _ in range(population_size)]
print(":: Initialized Population with best sneat agent checkpoint")

winning_streak = [0] * population_size # store the number of wins for this agent (including mutated ones)

# create the gym environment, and seed it
env = gym.make("SlimeVolley-v0")
env.seed(random_seed)
np.random.seed(random_seed)


history = []

from slimevolleygym import BaselinePolicy
# best_ga_json = "../training_scripts/ga_selfplay/ga_00135000.json"
# policy_ga = NEATPolicy(best_ga_json)
policy_base = BaselinePolicy()

for tournament in range(1, total_tournaments+1):
  
  # Random Pick Two Agents from Population
  left_idx, right_idx = np.random.choice(population_size, 2, replace=False)

  policy_right = NeatPolicy(population[right_idx], game)
  policy_left = NeatPolicy(population[left_idx], game)

  # Match between two agents
  raw_score_right, time_right = rollout(env, policy_right, policy_base)
  raw_score_left, time_left = rollout(env, policy_left, policy_base)
  
  # Faster win is better, slower lose is better
  score_right = raw_score_right + (1000 - time_right if raw_score_right > 0 else time_right) / 1000
  score_left = raw_score_left + (1000 - time_left if raw_score_left > 0 else time_left) / 1000
  
  length = (time_right + time_left) / 2
  history.append(int(length))
  
  # if score is positive, it means policy_right won.
  # win -> mutate, therefore winning streak is used as heuristic for generation number here
  if score_right == score_left: # if the game is tied, add noise to the left agent
    population[left_idx] = mutate(population[left_idx], p=hyp)
  elif score_right > score_left:
    population[left_idx] = mutate(population[right_idx], p=hyp)
    winning_streak[left_idx] = winning_streak[right_idx]
    winning_streak[right_idx] += 1
  else:
    population[right_idx] = mutate(population[left_idx], p=hyp)
    winning_streak[right_idx] = winning_streak[left_idx]
    winning_streak[left_idx] += 1

  if tournament % save_freq == 0:
    
    model_filename = os.path.join(logdir, "sneat_"+str(tournament).zfill(8)+".json")
    with open(model_filename, 'wt') as out: # save best solution
      record_holder = np.argmax(winning_streak)
      record = winning_streak[record_holder]
      population[record_holder].save(model_filename)

  if (tournament ) % 100 == 0: # print best solution
    record_holder = np.argmax(winning_streak)
    fig, _ = viewInd(population[record_holder])
    plt.close(fig)
    img = fig2img(fig)
    img.save(os.path.join(visdir, "sneat_"+str(tournament).zfill(8)+".png"))
    
    record = winning_streak[record_holder]
    print("tournament:", tournament,
          "best_winning_streak:", record,
          "mean_duration", np.mean(history),
          "stdev:", np.std(history),
         )
    history = []