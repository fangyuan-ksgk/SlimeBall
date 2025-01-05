# Trains an agent from scratch (no existing AI) using evolution
# GA with no cross-over, just mutation, and random tournament selection
# Not optimized for speed, and just uses a single CPU (mainly for simplicity)


# The issue with the mechanism: 
# Assume A has N winning streak, A>B in tournament. 
# In the next tournament, A (N+1) compete with A's mutation (N)
# - If A wins, A has (N+2) and A gives another mutation with (N+1) winning streak
# - If a loses, A's mutation grows to (N+1) and A's mutation gives offspring to replace A with (N) winning streak
# The issue is that no matter what, minimum winning streak do not decrease. 
# On gloabl population, tournament infect candidate and cause non-decreasing minimum winning streak
# It's like a virus.
# Increase of winning streak is guaranteed regardless of "mutation quality" or "actual quality of solution"
# -- That makes the genetic algorithm bad. 


import os
import json
import numpy as np
import gym
import slimevolleygym
import slimevolleygym.mlp as mlp
from slimevolleygym.mlp import games as games
from slimevolleygym.mlp import Model
from slimevolleygym import multiagent_rollout as rollout


# Settings
random_seed = 612
population_size = 128
total_tournaments = 500000
save_freq = 1000


# Log results
logdir = "ga_selfplay"
if not os.path.exists(logdir):
  os.makedirs(logdir)

def mutate(param, param_count):
  return param + np.random.normal(size=param_count) * 0.1

# Create two instances of a feed forward policy we may need later.
policy_left = Model(games['slimevolleylite']) # observation -> action model with specific network structure
policy_right = Model(games['slimevolleylite'])
param_count = policy_left.param_count
print("Number of parameters of the neural net policy:", param_count) # 273 for slimevolleylite

# store our population here
# FY: Each individual is the set of parameters for the policy network | "Population Size" number of policy network parameters
population = np.random.normal(size=(population_size, param_count)) * 0.5 # each row is an agent.
winning_streak = [0] * population_size # store the number of wins for this agent (including mutated ones)

# create the gym environment, and seed it
env = gym.make("SlimeVolley-v0")
env.seed(random_seed)
np.random.seed(random_seed)

history = []
for tournament in range(1, total_tournaments+1):
  # FY: each tournament is a match between two agents, winner and loser decided
  #     the search space is so small that randomized parameters lead to better policy ... 
  
  # Random Pick Two Agents from Population
  m, n = np.random.choice(population_size, 2, replace=False)

  policy_left.set_model_params(population[m])
  policy_right.set_model_params(population[n])

  # the match between the mth and nth member of the population
  score, length = rollout(env, policy_right, policy_left)

  history.append(length)
  # if score is positive, it means policy_right won.
  if score == 0: # if the game is tied, add noise to the left agent. | FY: both survive & mutation
    population[m] = mutate(population[m], param_count)
  if score > 0:
    population[m] = mutate(population[n], param_count)
    winning_streak[m] = winning_streak[n] # FY: mutated offspring inherits parent's crediential (winning streak)
    winning_streak[n] += 1
  if score < 0:
    population[n] = mutate(population[m], param_count)
    winning_streak[n] = winning_streak[m]
    winning_streak[m] += 1

  if tournament % save_freq == 0:
    model_filename = os.path.join(logdir, "ga_"+str(tournament).zfill(8)+".json")
    with open(model_filename, 'wt') as out: # save best solution
      record_holder = np.argmax(winning_streak)
      record = winning_streak[record_holder]
      json.dump([population[record_holder].tolist(), record], out, sort_keys=True, indent=0, separators=(',', ': '))

  if (tournament ) % 100 == 0: # print best solution
    record_holder = np.argmax(winning_streak)
    record = winning_streak[record_holder]
    print("tournament:", tournament,
          "best_winning_streak:", record,
          "mean_duration", np.mean(history),
          "stdev:", np.std(history),
         )
    history = []