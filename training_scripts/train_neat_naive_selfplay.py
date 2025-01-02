"""
Trains an agent using a simplified NEAT implementation that:
1. Uses fixed topology matching the GA implementation
2. Only performs weight mutations (no structural mutations)
3. Uses tournament selection with winning streak tracking
4. Maintains direct parameter mapping like GA
"""

import os
import numpy as np
import gym
import slimevolleygym
from slimevolleygym import multiagent_rollout as rollout
from slimevolleygym.mlp import Model, games
import json
import pickle
import argparse
from tqdm import tqdm

# Settings (matching GA's settings exactly)
parser = argparse.ArgumentParser(description='Train agent with self-play using GA-like approach')
parser.add_argument('--output-dir', type=str, default='zoo/neat_sp', help='Directory to save checkpoints')
parser.add_argument('--checkpoint-freq', type=int, default=1000, help='Frequency of checkpointing')
parser.add_argument('--population-size', type=int, default=128, help='Population size')
parser.add_argument('--total-tournaments', type=int, default=500000, help='Total number of tournaments')
parser.add_argument('--random-seed', type=int, default=612, help='Random seed')
args = parser.parse_args()

random_seed = args.random_seed
population_size = args.population_size
total_tournaments = args.total_tournaments
save_freq = args.checkpoint_freq

# Set random seeds like GA
env = gym.make("SlimeVolley-v0")
env.seed(random_seed)
np.random.seed(random_seed)

# Create output directory
zoo_path = args.output_dir
if not os.path.exists(zoo_path):
    os.makedirs(zoo_path)

class Genome:
    """Simple genome class to hold network parameters"""
    def __init__(self, key=None, param_count=None):
        self.key = key
        self.fitness = 0
        self.winning_streak = 0
        self.weights = np.random.normal(size=param_count) * 0.5 if param_count else None

class FixedTopologyNEAT:
    """Simple implementation with fixed topology matching GA's MLP structure"""
    def __init__(self):
        self.model = Model(games['slimevolleylite'])  # Same model as GA
        self.param_count = self.model.param_count
        print(f"Number of parameters: {self.param_count}")  # Should be 273 like GA
        
    def create_genome(self, key=None):
        """Create a genome with fixed topology matching GA's MLP"""
        return Genome(key=key, param_count=self.param_count)
    
    def mutate_weights(self, genome):
        """Mutate weights using same approach as GA"""
        genome.weights += np.random.normal(size=self.param_count) * 0.1  # Same mutation as GA
        
    def get_policy(self, genome):
        """Convert genome to policy for evaluation"""
        policy = Model(games['slimevolleylite'])
        policy.set_model_params(genome.weights)
        return policy

def evaluate_match(env, policy1, policy2):
    """Evaluate a single match between two policies"""
    score, length = rollout(env, policy1, policy2)
    return score, length

def eval_genomes(genomes, config):
    """Evaluate genomes using tournament selection with direct score-based fitness"""
    history = []
    
    # Convert genomes to policies (using same 10x10 structure as GA)
    neat_impl = FixedTopologyNEAT()
    policies = {}
    for genome_id, genome in genomes:
        if not hasattr(genome, 'winning_streak'):
            genome.winning_streak = 0  # Initialize if not present
        policy = neat_impl.get_policy(genome)
        policies[genome_id] = policy
        
    # Run tournaments (3x population size matches)
    num_tournaments = len(genomes) * 3
    
    for _ in tqdm(range(num_tournaments), desc="Generation Tournaments"):
        # Random tournament selection
        idx1, idx2 = np.random.choice(len(genomes), 2, replace=False)
        genome_id1, genome1 = genomes[idx1]
        genome_id2, genome2 = genomes[idx2]
        
        policy1 = policies[genome_id1]
        policy2 = policies[genome_id2]
        
        score, length = evaluate_match(env, policy1, policy2)
        history.append(length)
        
        # Handle tournament outcomes exactly like GA
        if score == 0:  # Tie - only mutate left agent (exactly like GA)
            neat_impl.mutate_weights(genome1)
            # Don't update winning streaks on ties (matching GA)
            genome1.fitness = genome1.winning_streak
            genome2.fitness = genome2.winning_streak
        elif score > 0:  # policy2 won
            # First copy winner's weights to loser (exactly like GA)
            genome1.weights = genome2.weights.copy()
            # Then mutate winner's weights (exactly like GA)
            neat_impl.mutate_weights(genome2)
            # Update winning streaks exactly like GA
            genome1.winning_streak = genome2.winning_streak  # Inherit streak
            genome2.winning_streak += 1
            # Update fitness based on winning streak
            genome1.fitness = genome1.winning_streak
            genome2.fitness = genome2.winning_streak
            
        elif score < 0:  # policy1 won
            # First copy winner's weights to loser (exactly like GA)
            genome2.weights = genome1.weights.copy()
            # Then mutate winner's weights (exactly like GA)
            neat_impl.mutate_weights(genome1)
            # Update winning streaks exactly like GA
            genome2.winning_streak = genome1.winning_streak  # Inherit streak
            genome1.winning_streak += 1
            # Update fitness based on winning streak
            genome1.fitness = genome1.winning_streak
            genome2.fitness = genome2.winning_streak
            
    return history

# Initialize population with GA-like parameters
best_path = "neat_naive_{gen}_best.pkl"

# Initialize NEAT implementation and population (exactly like GA)
neat_impl = FixedTopologyNEAT()
population = []
for _ in range(population_size):
    genome = neat_impl.create_genome()
    genome.winning_streak = 0  # Initialize winning streak
    genome.fitness = 0  # Initialize fitness
    population.append(genome)

# Main training loop (matching GA's tournament approach exactly)
history = []
for tournament in range(1, total_tournaments + 1):
    # Random tournament selection (exactly like GA)
    m, n = np.random.choice(population_size, 2, replace=False)
    genome_m = population[m]
    genome_n = population[n]
    
    # Get policies for evaluation
    policy_m = neat_impl.get_policy(genome_m)
    policy_n = neat_impl.get_policy(genome_n)
    
    # Evaluate match
    score, length = evaluate_match(env, policy_n, policy_m)  # Note: order matches GA
    history.append(length)
    
    # Handle tournament outcomes exactly like GA
    if score == 0:  # Tie - only mutate left agent
        neat_impl.mutate_weights(genome_m)
        # Don't update winning streaks on ties
        genome_m.fitness = genome_m.winning_streak
        genome_n.fitness = genome_n.winning_streak
    elif score > 0:  # policy_n won
        # First copy winner's weights to loser
        genome_m.weights = genome_n.weights.copy()
        # Then mutate winner's weights
        neat_impl.mutate_weights(genome_n)
        # Update winning streaks
        genome_m.winning_streak = genome_n.winning_streak
        genome_n.winning_streak += 1
        # Update fitness
        genome_m.fitness = genome_m.winning_streak
        genome_n.fitness = genome_n.winning_streak
    else:  # policy_m won
        # First copy winner's weights to loser
        genome_n.weights = genome_m.weights.copy()
        # Then mutate winner's weights
        neat_impl.mutate_weights(genome_m)
        # Update winning streaks
        genome_n.winning_streak = genome_m.winning_streak
        genome_m.winning_streak += 1
        # Update fitness
        genome_m.fitness = genome_m.winning_streak
        genome_n.fitness = genome_n.winning_streak
    
    if tournament % save_freq == 0:
        # Save best performing genome's weights in JSON format (compatible with Model class)
        best_genome = max(population, key=lambda g: g.fitness)
        model_filename = os.path.join(zoo_path, f"neat_naive_{tournament}_best.json")
        model_params = [best_genome.weights.tolist()]  # Model expects weights as first element in list
        with open(model_filename, 'w') as out:
            json.dump(model_params, out)
        
        # Print statistics (like GA)
        print(f"tournament: {tournament}",
              f"best_winning_streak: {best_genome.fitness}",
              f"mean_duration: {np.mean(history)}",
              f"stdev: {np.std(history)}")
        history = []

# Save final best genome's weights in JSON format
best_genome = max(population, key=lambda g: g.fitness)
model_filename = os.path.join(zoo_path, "neat_naive_final_best.json")
model_params = [best_genome.weights.tolist()]  # Model expects weights as first element in list
with open(model_filename, 'w') as out:
    json.dump(model_params, out)

print(f"Training complete. Final best winning streak: {best_genome.fitness}")
