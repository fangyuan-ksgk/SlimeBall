#!/usr/bin/env python
"""
Train an agent to play SlimeVolley using NEAT-Python.
"""
import os
import gym
import neat
import numpy as np
import slimevolleygym
from slimevolleygym import SlimeVolleyEnv, BaselinePolicy
from slimevolleygym.mlp import NEATPolicy
from slimevolleygym import multiagent_rollout as rollout
import pickle


config_path = "config-neat"

def eval_genomes(genomes: list[neat.DefaultGenome], config: neat.Config) -> None:
    """
    Evaluate all genomes in the population.
    NEAT policy v.s. Baseline policy
    In-place fitness assignment
    """
    env = gym.make('SlimeVolley-v0')

    opponent_policy = BaselinePolicy()
    for genome_id, genome in genomes:
        neat_policy = NEATPolicy(genome, config)
        score, length = rollout(env, neat_policy, opponent_policy)
        genome.fitness = score



# Load Configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

# Add these near your other reporter setup
num_generations = 3600 
generation_interval = 100
zoo_path = "../zoo/neat/"
if not os.path.exists(zoo_path):
    os.makedirs(zoo_path)

for gen in range(num_generations//generation_interval):
    winner = p.run(eval_genomes, generation_interval)
    with open(zoo_path + f'train_neat_{gen}.pkl', 'wb') as f:
        pickle.dump(winner, f)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# shutil copy config to zoo
import shutil
shutil.copy2(config_path, zoo_path + "config-neat")