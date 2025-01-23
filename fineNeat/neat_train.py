# Train NEAT with tournament-style reward assignment (Self-Play)
from fineNeat.neat_src import loadHyp, updateHyp
from fineNeat.domain import load_task
from fineNeat.neat_src import DataGatherer, Neat 
from fineNeat.domain.config import games
from fineNeat.domain.task_gym import GymTask
import numpy as np 
from tqdm import tqdm
import random
from time import time
from concurrent.futures import ProcessPoolExecutor
import os 
import gym
import argparse

game = games["slimevolley"]
task = GymTask(game)

hyp_default = 'fineNeat/p/default_sneat.json'
hyp_adjust = "fineNeat/p/volley_sparse.json"
fileName = "volley"

hyp = loadHyp(pFileName=hyp_default, load_task=load_task)
updateHyp(hyp,load_task,hyp_adjust)

neat = Neat(hyp)


def evaluate_individual(args):
    wMat, aVec = args
    return task.getFitness(wMat, aVec)

def parallelEval(pop, n_workers=None):
    if n_workers is None:
        n_workers = os.cpu_count()
    
    eval_args = [(ind.wMat, ind.aVec) for ind in pop]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = list(
            executor.map(evaluate_individual, eval_args)
        )
        rewards = np.array(futures, dtype=np.float64)
        
    return rewards

from fineNeat import DataGatherer
from tqdm import tqdm

LOG_PATH = "../runs/neat_non_sp/"
VIS_PATH = LOG_PATH + "/vis/"

if not os.path.exists(VIS_PATH):
    os.makedirs(VIS_PATH)

# Add before master()
def parse_args():
    parser = argparse.ArgumentParser(description='NEAT training script')
    parser.add_argument('--seed', type=int, default=612, help='Random seed')
    parser.add_argument('--save-freq', type=int, default=100, help='Save frequency')
    parser.add_argument('--hyp-default', type=str, default='fineNeat/p/default_sneat.json', help='Default hyperparameters file')
    parser.add_argument('--hyp-adjust', type=str, default='fineNeat/p/volley_sparse.json', help='Adjustment hyperparameters file')
    parser.add_argument('--logdir', type=str, default='../runs/neat_non_sp/', help='Log directory')
    return parser.parse_args()

def master():
    args = parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    
    # Update global variables based on args
    global hyp, LOG_PATH
    hyp = loadHyp(pFileName=args.hyp_default, load_task=load_task)
    updateHyp(hyp, load_task, args.hyp_adjust)
    LOG_PATH = args.logdir
    
    data = DataGatherer(fileName, hyp, LOG_PATH)
    neat = Neat(hyp)
  
    # better to start from a properly complicated network topology here
  
    save_freq = args.save_freq

    for gen in tqdm(range(hyp['maxGen']), total=hyp['maxGen'], desc="NEAT Generation Evolution"):        
        pop = neat.ask()            # Get newly evolved individuals from NEAT  
        reward = parallelEval(pop, n_workers=None)  # Send pop to be evaluated by workers
        neat.tell(reward)           # Send fitness to NEAT    

        data.gatherData(neat.pop, neat.species)
        print(gen, '\t - \t', data.display())

        if gen % save_freq == 0:
            if gen > 0: 
                grid_img = neat.printSpecies(neat.species, mute=True)
                grid_img.save(VIS_PATH + "neat_species_" + str(gen) + ".png")
            data.save(gen)

    # Clean up and data gathering at run end
    data.save(gen)
  
  
if __name__ == "__main__":
    master()