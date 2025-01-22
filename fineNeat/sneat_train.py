import os
import numpy as np
import gym
from slimevolleygym.mlp import games as games
from slimevolleygym.mlp import Model
from slimevolleygym import multiagent_rollout as rollout
from fineNeat import loadHyp, load_task, updateHyp, Ind, NeatPolicy, draw_img
import matplotlib.pyplot as plt
from tqdm import tqdm


# Settings
random_seed = 612
population_size = 128
total_tournaments = 1500000
save_freq = 1000

# Environment
# Hyperparameters
hyp_default = 'fineNeat/p/default_sneat.json'
# hyp_adjust = 'fineNeat/p/volley_default.json'
hyp_adjust = 'fineNeat/p/volley_sparse.json'

hyp = loadHyp(pFileName=hyp_default, load_task=load_task)
updateHyp(hyp,load_task,hyp_adjust)

# Log results
logdir = "../runs/sneat_sp_cons1"
visdir = "../runs/sneat_sp_cons1/vis"
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
population = [Ind.from_shapes([(game.input_size, 3), (3, game.output_size)]) for _ in range(population_size)]
winning_streak = [0] * population_size # store the number of wins for this agent (including mutated ones)

# create the gym environment, and seed it
env = gym.make("SlimeVolley-v0")
env.seed(random_seed)
np.random.seed(random_seed)


history = []

for tournament in tqdm(range(1, total_tournaments+1)):
  
  # Random Pick Two Agents from Population
  left_idx, right_idx = np.random.choice(population_size, 2, replace=False)

  policy_right = NeatPolicy(population[right_idx], game)
  policy_left = NeatPolicy(population[left_idx], game)

  # Match between two agents
  score, length = rollout(env, policy_right, policy_left)
  
  history.append(length)
  
  # if score is positive, it means policy_right won.
  # win -> mutate, therefore winning streak is used as heuristic for generation number here
  if score == 0: # if the game is tied, add noise to the left agent
    population[left_idx] = mutate(population[left_idx], p=hyp)
  elif score > 0:
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

  if (tournament ) % 1000 == 0: # print best solution
    record_holder = np.argmax(winning_streak)
    img = draw_img(population[record_holder])
    img.save(os.path.join(visdir, "sneat_"+str(tournament).zfill(8)+".png"))
    
    record = winning_streak[record_holder]
    print("tournament:", tournament,
          "best_winning_streak:", record,
          "mean_duration", np.mean(history),
          "stdev:", np.std(history),
         )
    history = []