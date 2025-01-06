# Train GA with Rotational Jacobian Estimation 

import json
import numpy as np
import gym, os
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

base_tournament = 50000

# Log results
logdir = "ga_jacobian_estimate2"
if not os.path.exists(logdir):
  os.makedirs(logdir)


def eval_parameter_fitness(params):
    policy_left.set_model_params(params)
    score, length = rollout(env, policy_left, policy_right)
    return score 

from jacobian_estimate import estimate_jacobian_dg
def mutate(param, param_count):
    # Get Jacobian estimate
    j = estimate_jacobian_dg(f=eval_parameter_fitness, x=param, num_samples=4)
    
    # Combine traditional random mutation with Jacobian-guided mutation
    jacobian_mutation = j * step_size
    
    return param + jacobian_mutation

policy_left = Model(games['slimevolleylite'])
policy_right = Model(games['slimevolleylite'])



from slimevolleygym import BaselinePolicy   
base_policy = BaselinePolicy()

param_count = policy_left.param_count
print("Number of parameters of the neural net policy:", param_count) # 273 for slimevolleylite


population = np.random.normal(size=(population_size, param_count)) * 0.5 # each row is an agent.

winning_streak = [0] * population_size # store the number of wins for this agent (including mutated ones)

# create the gym environment, and seed it
env = gym.make("SlimeVolley-v0")
env.seed(random_seed)
np.random.seed(random_seed)


step_size = 0.1
history = []

# Tournament selection and evolution
from tqdm import tqdm 
for tournament in tqdm(range(total_tournaments)):  # Changed from population_size to total_tournaments
  
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
    winning_streak[m] = winning_streak[n]
    winning_streak[n] += 1
  if score < 0:
    population[n] = mutate(population[m], param_count)
    winning_streak[n] = winning_streak[m]
    winning_streak[m] += 1
    
  if tournament % save_freq == 0:
    model_filename = os.path.join(logdir, "jacobian_"+str(tournament+base_tournament).zfill(8)+".json")
    with open(model_filename, 'wt') as out: # save best solution
      record_holder = np.argmax(winning_streak)
      record = winning_streak[record_holder]
      json.dump([population[record_holder].tolist(), record], out, sort_keys=True, indent=0, separators=(',', ': '))

  if (tournament ) % 100 == 0: # print best solution
    record_holder = np.argmax(winning_streak)
    record = winning_streak[record_holder]
    print("tournament:", tournament+base_tournament,
          "best_winning_streak:", record,
          "mean_duration", np.mean(history),
          "stdev:", np.std(history),
         )
    history = []