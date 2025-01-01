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
from slimevolleygym.mlp import NEATPolicy
from tqdm import tqdm
import pickle
import shutil
import matplotlib.pyplot as plt
from collections import defaultdict

# Settings
random_seed = 612
save_freq = 1
total_generations = 500

# Log results
logdir = "neat_selfplay"
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
# NEAT zoo path
zoo_path = "../zoo/neat_sp"
if not os.path.exists(zoo_path):
    os.makedirs(zoo_path)

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
config_name = "config-neat"
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    config_name)

# Create population and add reporters
pop = neat.Population(config)
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

class TrainingStats:
    def __init__(self):
        self.stats = defaultdict(list)
        self.generations = []
        self.current_fitnesses = []
    
    def record_generation(self, gen, pop, generation_stats):
        # Record generation number
        self.generations.append(gen)
        
        # Record fitness stats
        fitnesses = [g.fitness for g in pop.population.values()]
        self.current_fitnesses = fitnesses  # Store current fitness distribution
        self.stats['max_fitness'].append(max(fitnesses))
        self.stats['avg_fitness'].append(np.mean(fitnesses))
        
        # Record game duration stats
        self.stats['avg_duration'].append(np.mean(generation_stats))
        self.stats['std_duration'].append(np.std(generation_stats))
        
    def plot_stats(self, save_path):
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Fitness over time
        plt.subplot(2, 1, 1)
        plt.plot(self.generations, self.stats['max_fitness'], label='Max Fitness')
        plt.plot(self.generations, self.stats['avg_fitness'], label='Avg Fitness')
        plt.title('Fitness over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Current Population Fitness Histogram
        plt.subplot(2, 1, 2)
        plt.hist(self.current_fitnesses, bins=30, edgecolor='black')
        plt.title('Current Population Fitness Distribution')
        plt.xlabel('Fitness')
        plt.ylabel('Number of Individuals')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# After creating your NEAT population and before the main training loop:
training_stats = TrainingStats()

best_path = "zoo/neat_sp/neat_{gen}_best.pkl"

# Modified main loop to track and save statistics
generation_stats = []
for gen in range(total_generations):
    # Run one generation and get history directly
    current_history = eval_genomes(list(pop.population.items()), config)
    generation_stats.extend(current_history)
    
    if gen % save_freq == 0:
        # Record and plot statistics
        training_stats.record_generation(gen, pop, generation_stats)
        plot_path = os.path.join(logdir, f"training_progress.png")
        training_stats.plot_stats(plot_path)
        
        # Save best performing genome
        best_genome = max(pop.population.values(), key=lambda g: g.fitness)
        
        # Print statistics similar to GA version
        print(f"generation: {gen}",
              f"best_fitness: {best_genome.fitness}",
              f"mean_duration: {np.mean(generation_stats)}",
              f"stdev: {np.std(generation_stats)}")
        generation_stats = []
        
# Save best performing genome
best_genome = max(pop.population.values(), key=lambda g: g.fitness)
model_filename = os.path.join(zoo_path, best_path.format(gen=gen))
with open(model_filename, 'wb') as out:
    pickle.dump(best_genome, out)
    
# Copy config file to zoo 
config_filename = os.path.join(zoo_path, config_name)
shutil.copy(config_name, config_filename)