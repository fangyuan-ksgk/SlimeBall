#!/usr/bin/env python
"""
Train an agent to play SlimeVolley using NEAT-Python.
"""
import os
import gym
import neat
import numpy as np
import slimevolleygym
from slimevolleygym import SlimeVolleyEnv

def eval_genome(genome, config):
    """
    Evaluate a single genome against the baseline agent over multiple episodes.
    Returns the average fitness score based on cumulative reward, survival time,
    and win/loss ratio across all episodes.
    Uses the survival bonus mode for better training signal.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # Disable environment checker to avoid numpy bool type issues
    env = gym.make("SlimeVolley-v0", disable_env_checker=True)
    env.survival_bonus = True  # Enable survival bonus
    
    num_episodes = 3  # Number of episodes to evaluate each genome
    total_fitness = 0
    total_won_rounds = 0
    total_lost_rounds = 0
    
    for episode in range(num_episodes):
        episode_fitness = 0
        obs = env.reset()
        done = False
        timesteps = 0
        
        # Run the episode
        while not done:
            # Get action from neural network
            action_values = net.activate(obs)
            # Convert to binary actions (threshold at 0.5)
            action = [1 if v > 0.5 else 0 for v in action_values]
            
            # Take action in environment
            obs, reward, done, info = env.step(action)
            episode_fitness += reward
            timesteps += 1
            
            # Track game outcomes
            if 'ale.lives' in info and 'ale.otherLives' in info:
                if info['ale.lives'] > info['ale.otherLives']:
                    total_won_rounds += 1
                elif info['ale.lives'] < info['ale.otherLives']:
                    total_lost_rounds += 1
        
        total_fitness += episode_fitness
    
    env.close()
    
    # Calculate average fitness across episodes
    avg_fitness = total_fitness / num_episodes
    
    # Add win/loss ratio to fitness if any games were played
    if total_won_rounds + total_lost_rounds > 0:
        win_ratio = total_won_rounds / (total_won_rounds + total_lost_rounds)
        avg_fitness += win_ratio * 2  # Scale win ratio to be significant
    
    return avg_fitness

def eval_genomes(genomes, config):
    """
    Evaluate all genomes in the population.
    """
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run_neat(config_file):
    """
    Run NEAT algorithm to train an agent.
    """
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)

    # Create the population
    pop = neat.Population(config)

    # Add reporters to show progress
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    # Save checkpoint every 10 generations
    pop.add_reporter(neat.Checkpointer(10, filename_prefix='neat-checkpoint-'))

    # Run for up to 50 generations
    winner = pop.run(eval_genomes, 50)

    # Save the winner
    with open('winner-feedforward', 'wb') as f:
        import pickle
        pickle.dump(winner, f)

    print('\nBest genome:\n{!s}'.format(winner))
    return winner

if __name__ == '__main__':
    # Get local directory
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat')
    
    winner = run_neat(config_path)
