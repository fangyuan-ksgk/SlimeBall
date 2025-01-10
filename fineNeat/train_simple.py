import os
import json
import numpy as np
import gym
import slimevolleygym
import slimevolleygym.mlp as mlp
from slimevolleygym.mlp import games as games
from slimevolleygym.mlp import Model
from slimevolleygym import multiagent_rollout as rollout
from neat_src import Ind

# Settings
random_seed = 612
population_size = 128
total_tournaments = 5000
save_freq = 1000


# Log results
logdir = "../runs/sneat_sp"
if not os.path.exists(logdir):
  os.makedirs(logdir)

def mutate(ind): 
    child, _ = ind.mutate()
    if child is None: 
        child, _ = ind.mutate(mute_top_change=True)
    return child

game = games['slimevolleylite']
population = [Ind.from_shapes([(12, 3), (3, 2)]) for _ in range(population_size)]
winning_streak = [0] * population_size # store the number of wins for this agent (including mutated ones)

# create the gym environment, and seed it
env = gym.make("SlimeVolley-v0")
env.seed(random_seed)
np.random.seed(random_seed)

history = []
for tournament in range(1, total_tournaments+1):
  
  # Random Pick Two Agents from Population
  left_idx, right_idx = np.random.choice(population_size, 2, replace=False)
  
  policy_right = Model.from_indiv(population[right_idx], game)
  policy_left = Model.from_indiv(population[left_idx], game)

  # Match between two agents
  score, length = rollout(env, policy_right, policy_left)
  
  history.append(length)
  
  # if score is positive, it means policy_right won.
  if score == 0: # if the game is tied, add noise to the left agent
    population[left_idx] = mutate(population[left_idx])
  elif score > 0:
    population[left_idx] = mutate(population[right_idx])
    winning_streak[left_idx] = winning_streak[right_idx]
    winning_streak[right_idx] += 1
  else:
    population[right_idx] = mutate(population[left_idx])
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
    record = winning_streak[record_holder]
    print("tournament:", tournament,
          "best_winning_streak:", record,
          "mean_duration", np.mean(history),
          "stdev:", np.std(history),
         )
    history = []