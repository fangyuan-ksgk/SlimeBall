import numpy as np
from ..utils import *

class Species():
  """Species class, only contains fields: all methods belong to the NEAT class.
  Note: All 'species' related functions are part of the Neat class, though defined in this file.
  """

  def __init__(self,seed):
    """Intialize species around a seed
    Args:
      seed - (Ind) - individual which anchors seed in compatibility space

    Attributes:
      seed       - (Ind)   - individual who acts center of species
      members    - [Ind]   - individuals in species
      bestInd    - (Ind)   - highest fitness individual ever found in species
      bestFit    - (float) - highest fitness ever found in species
      lastImp    - (int)   - generations since a new best individual was found
      nOffspring - (int)   - new individuals to create this generation
    """
    self.seed = seed      # Seed is type Ind
    self.members = [seed] # All inds in species
    self.bestInd = seed
    self.bestFit = seed.fitness
    self.lastImp = 0
    self.nOffspring = []

def speciate(self):  
  # Readbility
  p = self.p # algorithm hyperparameters
  pop = self.pop # population
  species = self.species # species

  if p['alg_speciate'] == 'neat':
    # Adjust species threshold to track desired number of species
    if len(species) > p['spec_target']: # increase threshold to decrease number of species
      p['spec_thresh'] += p['spec_compatMod']

    if len(species) < p['spec_target']: # decrease threshold to increase number of species
      p['spec_thresh'] -= p['spec_compatMod']

    if p['spec_thresh'] < p['spec_threshMin']: # not too small threshold, otherwise species are the same
      p['spec_thresh'] = p['spec_threshMin']

    species, pop = self.assignSpecies  (pop, p)
    species      = self.assignOffspring(species, pop, p)

  elif p['alg_speciate'] == "none": 
    # Recombination takes a species, when there is no species we dump the whole population into one species that is awarded all offspring
    species = [Species(pop[0])]
    species[0].nOffspring = p['popSize']
    for ind in pop:
      ind.species = 0
    species[0].members = pop

  # Update
  self.p = p
  self.pop = pop
  self.species = species

def assignSpecies(self, pop, p):
  """ 
  Does this decreate number of species if threshold is reached ?? 
  """
  
  species = [Species(pop[0])]
  species[0].nOffspring = p['popSize']

  # Assign members of population to first species within compat distance
  for i in range(len(pop)):
    assigned = False
    for iSpec in range(len(species)):
      ref = np.copy(species[iSpec].seed.conn)
      ind = np.copy(pop[i].conn)
      cDist = self.compatDist(ref,ind)
      if cDist < p['spec_thresh']:
        pop[i].species = iSpec
        species[iSpec].members.append(pop[i])
        assigned = True
        break

    # If no seed is close enough, start your own species
    if not assigned:
      pop[i].species = iSpec+1
      species.append(Species(pop[i]))

  return species, pop

def assignOffspring(self, species, pop, p):
  
  nSpecies = len(species)
  if nSpecies == 1:
    species[0].offspring = p['popSize']
  else:
    # -- Fitness Sharing
    # Rank all individuals
    popFit = np.asarray([ind.fitness for ind in pop])
    popRank = tiedRank(popFit) # fitter individuals get higher rank score
    if p['select_rankWeight'] == 'exp':
      rankScore = 1/popRank
    elif p['select_rankWeight'] == 'lin':
      rankScore = 1+abs(popRank-len(popRank))
    else:
      print("Invalid rank weighting (using linear)")
      rankScore = 1+abs(popRank-len(popRank))
      
    specId = np.asarray([ind.species for ind in pop])

    # Best and Average Fitness of Each Species
    speciesFit = np.zeros((nSpecies,1))
    speciesTop = np.zeros((nSpecies,1))
    for iSpec in range(nSpecies):
      if not np.any(specId==iSpec):
        speciesFit[iSpec] = 0
      else:
        speciesFit[iSpec] = np.mean(rankScore[specId==iSpec])
        speciesTop[iSpec] = np.max(popFit[specId==iSpec])

        # Did the species improve?
        if speciesTop[iSpec] > species[iSpec].bestFit:
          species[iSpec].bestFit = speciesTop[iSpec]
          bestId = np.argmax(popFit[specId==iSpec])
          species[iSpec].bestInd = species[iSpec].members[bestId]
          species[iSpec].lastImp = 0
        else:
          species[iSpec].lastImp += 1

        # Stagnant species don't recieve species fitness
        if species[iSpec].lastImp > p['spec_dropOffAge']:
          speciesFit[iSpec] = 0
          
    # -- Assign Offspring
    if sum(speciesFit) == 0:
      speciesFit = np.ones((nSpecies,1))
      print("WARN: Entire population stagnant, continuing without extinction")
      
    offspring = bestIntSplit(speciesFit, p['popSize']) # assign offspring proportionally to species fitness
    for iSpec in range(nSpecies):
      species[iSpec].nOffspring = offspring[iSpec]
      
  # Extinction    
  species[:] = [s for s in species if s.nOffspring != 0]

  return species

from matplotlib import pyplot as plt 
from ..vis.viewInd import viewInd, fig2img 
from PIL import Image
def printSpecies(self, species, mute: bool = False): 
  if not mute: 
    print(" :: Total of species: ", len(species))
  spec_nets = []
  for spec_idx, spec in enumerate(species): 
    if not mute: 
      print(" :: Species: ", spec_idx, " :: Offspring: ", spec.nOffspring)
    spec.seed.express()
    # Make individual visualizations larger with 4:3 ratio
    fig, ax = viewInd(spec.seed)
    # Remove title
    ax.set_title('')
    # Add species info as text with larger font size
    plt.text(0.45, 0.98, f'Species {spec_idx}\nOffspring: {spec.nOffspring}', 
             transform=ax.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top',
             fontsize=16,  # Increased font size
             fontweight='bold')  # Made text bold for better visibility
    img = fig2img(fig)
    # Resize the image to 800x600 (4:3 ratio)
    img = img.resize((800, 600))
    spec_nets.append(img)
    plt.close(fig)

  # Calculate dimensions for the final image
  n_cols = min(3, len(species))
  n_rows = len(species) // n_cols + (1 if len(species) % n_cols > 0 else 0)
  
  # Create blank image for the grid
  grid_width = n_cols * 800
  grid_height = n_rows * 600
  grid_img = Image.new('RGB', (grid_width, grid_height), 'white')
  
  # Paste images into grid
  for i, img in enumerate(spec_nets):
    x = (i % n_cols) * 800
    y = (i // n_cols) * 600
    grid_img.paste(img, (x, y))
    
  return grid_img
      

def compatDist(self, ref, ind):
  
  # Find matching genes
  IA, IB = quickINTersect(ind[0,:].astype(int),ref[0,:].astype(int))          
  
  # Calculate raw genome distances
  ind[3,np.isnan(ind[3,:])] = 0
  ref[3,np.isnan(ref[3,:])] = 0
  weightDiff = abs(ind[3,IA] - ref[3,IB])
  geneDiff   = sum(np.invert(IA)) + sum(np.invert(IB)) # |A + B| - |A ∩ B|

  # Normalize and take weighted sum
  nInitial = self.p['ann_nInput'] + self.p['ann_nOutput']
  longestGenome = max(len(IA),len(IB)) - nInitial
  weightDiff = np.mean(weightDiff)
  geneDiff   = geneDiff   / (1+longestGenome)

  dist = geneDiff   * self.p['spec_geneCoef']      \
       + weightDiff * self.p['spec_weightCoef']  
  return dist
