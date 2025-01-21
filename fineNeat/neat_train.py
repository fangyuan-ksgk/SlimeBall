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

game = games["slimevolley"]
task = GymTask(game)

hyp_default = 'fineNeat/p/default_sneat.json'
hyp_adjust = "fineNeat/p/volley_sparse.json"
fileName = "volley"

hyp = loadHyp(pFileName=hyp_default, load_task=load_task)
updateHyp(hyp,load_task,hyp_adjust)
print('\t*** Running with hyperparameters: ', hyp_adjust, '\t***')

neat = Neat(hyp)

# Parallal Reward Assignment (Speed-Up 3x)
def evaluate_individual(args):
    ind, pop, n_opponents = args
    return task.getTournamentScore(ind, random.sample(pop, n_opponents))

def parallelEval(pop, task, n_opponents=6, n_workers=None):
    if n_workers is None:
        n_workers = os.cpu_count()
    
    eval_args = [(ind, pop, n_opponents) for ind in pop]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = list(
            executor.map(evaluate_individual, eval_args)
        )
        rewards = np.array(futures, dtype=np.float64)
        
    return rewards

from fineNeat import DataGatherer
from tqdm import tqdm

LOG_PATH = "../runs/neat/"
VIS_PATH = LOG_PATH + "/vis/"

if not os.path.exists(VIS_PATH):
    os.makedirs(VIS_PATH)

# -- Run NEAT ------------------------------------------------------------ -- #
def master(): 
  """Main NEAT optimization script
  """
  global fileName, hyp
  data = DataGatherer(fileName, hyp, LOG_PATH)
  neat = Neat(hyp)
  
  # better to start from a properly complicated network topology here
  
  save_freq = 100

  for gen in tqdm(range(hyp['maxGen']), total=hyp['maxGen'], desc="NEAT Generation Evolution"):        
    pop = neat.ask()            # Get newly evolved individuals from NEAT  
    reward = parallelEval(pop, task, n_opponents=6, n_workers=None)  # Send pop to be evaluated by workers
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