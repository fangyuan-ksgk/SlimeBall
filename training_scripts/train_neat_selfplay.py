# Trains an agent from scratch (no existing AI) using evolution
# NEAT GA with crossover and mutation, no recurrent connections
# Not optimized for speed, and just uses a single CPU (mainly for simplicity)

import os
import json
import numpy as np
import gym
import neat
import slimevolleygym

# Settings
random_seed = 612
save_freq = 1000
total_generations = 500

# Log results
logdir = "neat_selfplay"
if not os.path.exists(logdir):
    os.makedirs(logdir)

def eval_genomes(genomes, config):
    """Evaluate genomes in self-play tournaments"""
    # Create environment once per generation
    env = gym.make("SlimeVolley-v0")
    
    for i, (genome_id1, genome1) in enumerate(genomes[:-1]):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        
        for genome_id2, genome2 in genomes[i+1:]:
            net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
            
            # Play one game (genome1 vs genome2)
            obs = env.reset()
            done = False
            while not done:
                action1 = np.argmax(net1.activate(obs))
                obs_other = env.obs_agent_two()
                action2 = np.argmax(net2.activate(obs_other))
                
                obs, reward, done, _ = env.step([action1, action2])
            
            # Update fitness based on game outcome
            if reward > 0:  # genome1 won
                genome1.fitness += 1
            elif reward < 0:  # genome2 won
                genome2.fitness += 1
            # Ties result in no fitness change

# Load NEAT configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    'config-feedforward')

# Create population and add reporters
pop = neat.Population(config)
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

# Run evolution
winner = pop.run(eval_genomes, total_generations)

# Save winner
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
model_filename = os.path.join(logdir, "neat_winner.json")
with open(model_filename, 'wt') as out:
    json.dump([winner.fitness, winner.size()], out)