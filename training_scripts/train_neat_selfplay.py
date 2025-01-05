#!/usr/bin/env python
"""
Train an agent to play SlimeVolley using NEAT-Python.
"""
import os
import gym
import neat
import numpy as np
import pickle
import slimevolleygym
from slimevolleygym import SlimeVolleyEnv, BaselinePolicy
from slimevolleygym.mlp import NEATPolicy
from slimevolleygym import multiagent_rollout as rollout

class RandomPolicy:
  def __init__(self):
    self.action_space = gym.spaces.MultiBinary(3)
    pass
  def predict(self, obs):
    return self.action_space.sample()


config_path = "config-neat"

def create_eval_function(opponent_genome, config):
    """Creates an eval_genomes function with a fixed opponent"""
    def eval_genomes(genomes: list[neat.DefaultGenome], config: neat.Config) -> None:
        env = gym.make('SlimeVolley-v0')
        
        # Use baseline policy if no opponent genome provided
        opponent_policy = RandomPolicy() if opponent_genome is None else NEATPolicy(opponent_genome, config)
        
        for genome_id, genome in genomes:
            neat_policy = NEATPolicy(genome, config)
            score, length = rollout(env, neat_policy, opponent_policy)
            genome.fitness = score
    
    return eval_genomes

# Load Configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

# Run evolution with self-play
num_generations = 10
num_populations = 360
opponent_genome = None  # Start with None to use baseline policy

zoo_path = "../zoo/neat_sp/"
if not os.path.exists(zoo_path):
    os.makedirs(zoo_path)
        
for pop in range(num_populations):
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(False))
    eval_function = create_eval_function(opponent_genome, config)
    winner = p.run(eval_function, num_generations)
    opponent_genome = winner
    
    with open(zoo_path + f'train_neat_selfplay_{pop}.pkl', 'wb') as f:
        pickle.dump(winner, f)
        
# shutil copy config to zoo
import shutil
shutil.copy2(config_path, zoo_path + "config-neat")