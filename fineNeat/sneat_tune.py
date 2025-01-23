# Evolve from best sneat agent to beat baseline
# - through evaluating against miscellaneous opponents
# - as an experiment, I'd like to see if evolve sneat agent to beat baseline works ... 
# - essentially we'd keep mutating it until it beats baseline 

import os
import numpy as np
import gym
import argparse
from slimevolleygym.mlp import games as games
from slimevolleygym.mlp import Model
from slimevolleygym import multiagent_rollout as rollout
from slimevolleygym import BaselinePolicy
from fineNeat import loadHyp, load_task, updateHyp, Ind, NeatPolicy, viewInd, fig2img, NEATPolicy
from fineNeat.neat_src.jacobian import estimate_jacobian_dg
import matplotlib.pyplot as plt
from tqdm import tqdm

# What if one choose to fix topology and just tune the 'connection weights'? 
# -- that's just conn[3,:] right? do we have a clever way to do this ? 

class RandomPolicy:
  def __init__(self, path):
    self.action_space = gym.spaces.MultiBinary(3)
    pass
  def predict(self, obs):
    return self.action_space.sample()
  

def schedule_mutate(tournament: int, topo_mut_freq: float = 1/16, period_length: int = 160000) -> bool:
    if tournament % period_length < int(topo_mut_freq * period_length):
        return True 
    else:
        return False

def mutate(ind, p, tournament, topo_mut_freq: float = 1/16, period_length: int = 160000): 
    if schedule_mutate(tournament, topo_mut_freq, period_length):
        child, _ = ind.mutate(p=p)
        if child: 
            return child 
    return ind.safe_mutate(p)


def connW_to_policy(connW, connG, nodeG, game=games['slimevolleylite']): 
    connG[3,:] = connW 
    ind = Ind(nodeG, connG)
    policy = NeatPolicy(ind, game)
    return policy
  
def connW_to_ind(connW, connG, nodeG): 
    connG[3,:] = connW 
    ind = Ind(nodeG, connG)
    return ind
    
def eval_ind_parameter_fitness(connW, connG, nodeG, opponent_policy, env):
    max_time = 3000
    policy_left = connW_to_policy(connW, connG, nodeG)
    raw_score, _time = rollout(env, policy_left, opponent_policy)
    score = raw_score * (1 - _time / max_time)
    return score 
  
def jacobian_step(ind_left, ind_right, step_size, env, game=games['slimevolleylite']):
    policy_right = NeatPolicy(ind_right, game)
    connW_left = ind_left.connW[3,:]
    eval_parameter_fitness = lambda connW: eval_ind_parameter_fitness(connW, ind_left.connG, ind_left.nodeG, policy_right, env)
          
    # Get Jacobian estimate
    j = estimate_jacobian_dg(f=eval_parameter_fitness, x=connW_left, num_samples=4)
    
    # Combine traditional random mutation with Jacobian-guided mutation
    jacobian_mutation = j * step_size
    
    # Apply mutation
    connW_mutated = connW_left + jacobian_mutation
    ind_mutated = connW_to_ind(connW_mutated, np.copy(ind_left.connG), np.copy(ind_left.nodeG))
    
    return ind_mutated

def eval(env, policy_left, policy_right, base_policy, villain_policy, selfplay=False):
    """ 
    Unify self-play with non-selfplay -- both involves two agents playing against an opponent policy
    """
    max_time = 3000
    if selfplay: # switch to play against villain policy 
        opponent_policy = villain_policy
    else:
        opponent_policy = base_policy     
        
    raw_score_right, time_right = rollout(env, policy_right, opponent_policy)
    raw_score_left, time_left = rollout(env, policy_left, opponent_policy)
    score_right = raw_score_right * (1 - time_right / max_time)
    score_left = raw_score_left * (1 - time_left / max_time)
    return score_right, score_left, (time_right + time_left) / 2
  
  
def update_winning_streak(winning_streak, left_idx, right_idx, score_right, score_left, mut_discount=0.8):
    """ 
    Simple approach: Absolute score to represent quality
    """
    if score_right > score_left: 
      winning_streak[left_idx] = winning_streak[right_idx] * mut_discount
      winning_streak[right_idx] = score_right
    elif score_right < score_left: 
      winning_streak[right_idx] = winning_streak[left_idx] * mut_discount
      winning_streak[left_idx] = score_left
    else: 
      winning_streak[left_idx] = score_left * mut_discount

    return winning_streak
  
  
def pick_villain(prev_villain_policy, tournament: int, population, winning_streak, period: int = 1000):
    """ 
    Pick a new villain policy from the population
    w.p. select random policy
    """
    if prev_villain_policy is None or tournament % period == 0:
        villain_idx = np.random.choice(np.flatnonzero(winning_streak == np.max(winning_streak)))
        villain = population[villain_idx]
        winning_streak[:] = [0] * len(winning_streak) # reset winning streak
        return NeatPolicy(villain, games['slimevolleylite']), winning_streak
    return prev_villain_policy, winning_streak
  
  
def main(args):
    # Initialize hyperparameters
    hyp = loadHyp(pFileName=args.hyp_default, load_task=load_task)
    updateHyp(hyp, load_task, args.hyp_adjust)

    # Create directories
    logdir = args.logdir
    visdir = os.path.join(logdir, "vis")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(visdir, exist_ok=True)

    game = games['slimevolleylite']
    load_ind = Ind.load(args.checkpoint)  
    population = [load_ind.safe_mutate(p=hyp) for _ in range(args.population_size)]
    print(":: Initialized Population with best sneat agent checkpoint")

    winning_streak = [0] * args.population_size

    # Setup environment
    env = gym.make("SlimeVolley-v0")
    env.seed(args.seed)
    np.random.seed(args.seed)
    policy_base = BaselinePolicy()
    villain_policy = None

    history = []

    for tournament in tqdm(range(1, args.total_tournaments+1)):
        left_idx, right_idx = np.random.choice(args.population_size, 2, replace=False)

        policy_right = NeatPolicy(population[right_idx], game)
        policy_left = NeatPolicy(population[left_idx], game)
        
        villain_policy, winning_streak = pick_villain(villain_policy, tournament, population, winning_streak, period=args.period)
        
        score_right, score_left, length = eval(env, policy_left, policy_right, policy_base, villain_policy, selfplay=args.selfplay)
        history.append(int(length))
        

        # Tournament based population update 
        if score_right == score_left:
            population[left_idx] = mutate(population[left_idx], p=hyp, tournament=tournament, topo_mut_freq=args.topo_mut_freq)
        elif score_right > score_left:
            population[left_idx] = mutate(population[right_idx], p=hyp, tournament=tournament, topo_mut_freq=args.topo_mut_freq)
        else:
            population[right_idx] = mutate(population[left_idx], p=hyp, tournament=tournament, topo_mut_freq=args.topo_mut_freq)
            
        winning_streak = update_winning_streak(winning_streak, 
                                               left_idx, 
                                               right_idx, 
                                               score_right, 
                                               score_left, 
                                               mut_discount=args.mut_discount)

        if tournament % args.save_freq == 0:
            model_filename = os.path.join(logdir, f"sneat_{tournament:08d}.json")
            with open(model_filename, 'wt') as out:
                record_holder = np.argmax(winning_streak)
                population[record_holder].save(model_filename)

        if (tournament) % args.save_freq == 0:
            record_holder = np.argmax(winning_streak)
            fig, _ = viewInd(population[record_holder])
            plt.close(fig)
            img = fig2img(fig)
            img.save(os.path.join(visdir, f"sneat_{tournament:08d}.png"))
            
            record = winning_streak[record_holder]
            print(f"tournament: {tournament}, best_winning_streak: {record}, "
                  f"mean_duration: {np.mean(history)}, stdev: {np.std(history)}")
            history = []

    info_str = f"Load from file: {args.checkpoint}" + "\n" + f"Population size: {args.population_size}" + "\n" + f"Total tournaments: {args.total_tournaments}" + "\n" + f"Save frequency: {args.save_freq}" + "\n" + f"Hyperparameters: {args.hyp_default} and {args.hyp_adjust}"
    print(info_str)
    with open(os.path.join(logdir, "info.txt"), 'wt') as out:
        out.write(info_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SNEAT tuning script')
    parser.add_argument('--seed', type=int, default=612, help='Random seed')
    parser.add_argument('--population-size', type=int, default=128, help='Population size')
    parser.add_argument('--total-tournaments', type=int, default=120000, help='Total number of tournaments')
    parser.add_argument('--save-freq', type=int, default=1000, help='Save frequency')
    parser.add_argument('--hyp-default', type=str, default='fineNeat/p/default_sneat.json', help='Default hyperparameters file')
    parser.add_argument('--hyp-adjust', type=str, default='fineNeat/p/volley_sparse.json', help='Adjustment hyperparameters file')
    parser.add_argument('--logdir', type=str, default='../runs/sneat_tune_base', help='Log directory')
    parser.add_argument('--checkpoint', type=str, default='../zoo/sneat_check/sneat_00360000_small.json', help='Checkpoint file to start from')
    parser.add_argument("--selfplay", action="store_true", help="Use selfplay")
    parser.add_argument("--period", type=int, default=1000, help="Period for picking a new villain")
    parser.add_argument("--topo-mut-freq", type=float, default=1/16, help="Frequency of topology mutation")
    parser.add_argument("--mut-discount", type=float, default=0.8, help="Mutation discount factor")
    args = parser.parse_args()
    main(args)
