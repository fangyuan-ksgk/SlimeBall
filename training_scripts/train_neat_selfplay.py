# Trains an agent from scratch (no existing AI) using evolution
# NEAT GA with crossover and mutation, no recurrent connections
# Not optimized for speed, and just uses a single CPU (mainly for simplicity)

import os
import json
import numpy as np
import gym
import neat
import slimevolleygym
from slimevolleygym import multiagent_rollout as rollout
from tqdm import tqdm

# Settings
random_seed = 612
save_freq = 1
total_generations = 500

# Log results
logdir = "neat_selfplay"
if not os.path.exists(logdir):
    os.makedirs(logdir)

class NEATPolicy:
    """Policy wrapper for NEAT neural networks"""
    def __init__(self, net):
        self.net = net
        self.winning_streak = 0
    
    def predict(self, obs):
        """Returns action in the format expected by the environment"""
        # Convert network output to a 3-element action array
        # where each element is either 0 or 1
        output = self.net.activate(obs)
        return [1 if o > 0 else 0 for o in output]

def evaluate_match(env, policy1, policy2):
    """Evaluate a single match between two policies using the same rollout as GA version"""
    score, length = rollout(env, policy1, policy2)
    return score, length

def eval_genomes(genomes, config):
    """Evaluate genomes using random tournament selection like in GA version"""
    env = gym.make("SlimeVolley-v0")
    history = []
    
    # Convert genomes to policies
    policies = {}
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        policies[genome_id] = NEATPolicy(net)
        genome.fitness = 0  # Reset fitness
    
    # Run random tournaments instead of round-robin
    num_tournaments = len(genomes) * 2  # Each genome plays ~4 matches on average
    
    # Add progress bar
    for _ in tqdm(range(num_tournaments), desc=f"Generation Tournaments", leave=False):
        # Randomly select two different genomes
        idx1, idx2 = np.random.choice(len(genomes), 2, replace=False)
        genome_id1, genome1 = genomes[idx1]
        genome_id2, genome2 = genomes[idx2]
        
        policy1 = policies[genome_id1]
        policy2 = policies[genome_id2]
        
        score, length = evaluate_match(env, policy1, policy2)
        history.append(length)
        
        # Update fitness
        if score > 0:  # policy2 won
            genome2.fitness += 1
            policy2.winning_streak += 1
            genome1.fitness -= 1
        elif score < 0:  # policy1 won
            genome1.fitness += 1
            policy1.winning_streak += 1
            genome2.fitness -= 1
            
    return history

# Load NEAT configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    'config-neat')

# Create population and add reporters
pop = neat.Population(config)
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

# Modified main loop to track and save statistics
generation_stats = []
for gen in range(total_generations):
    # Run one generation and get history directly
    current_history = eval_genomes(list(pop.population.items()), config)
    generation_stats.extend(current_history)
    
    if gen % save_freq == 0:
        # Save best performing genome
        best_genome = max(pop.population.values(), key=lambda g: g.fitness)
        model_filename = os.path.join(logdir, f"neat_{str(gen).zfill(8)}.json")
        with open(model_filename, 'wt') as out:
            json.dump([best_genome.fitness, best_genome.size()], out)
        
        # Print statistics similar to GA version
        print(f"generation: {gen}",
              f"best_fitness: {best_genome.fitness}",
              f"mean_duration: {np.mean(generation_stats)}",
              f"stdev: {np.std(generation_stats)}")
        generation_stats = []