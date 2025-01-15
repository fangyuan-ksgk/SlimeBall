import os
import numpy as np
import copy
from .ann import exportNet

    
class DataGatherer():
  """Data recorder for NEAT algorithm
  """
  def __init__(self, filename, hyp, log_path = "../runs/neat/"): 
    """
    Args:
      filename - (string) - path+prefix of file output destination
      hyp      - (dict)   - algorithm hyperparameters
    """
    self.filename = filename # File name path + prefix
    self.p = hyp
    
    # Initialize empty fields
    self.elite = []
    self.best = []
    self.bestFitVec = []
    self.spec_fit = []
    self.field = ['x_scale','fit_med','fit_max','fit_top',\
                  'node_med','conn_med',\
                  'elite','best']

    if self.p['alg_probMoo'] > 0:
      self.objVals = np.array([])

    for f in self.field[:-2]:
      exec('self.' + f + ' = np.array([])')
      #e.g. self.fit_max   = np.array([]) 

    self.newBest = False
    
    self.reset_log_path(log_path)
    
  def reset_log_path(self, LOG_PATH): 
    os.makedirs(LOG_PATH, exist_ok=True)
    VIS_PATH = LOG_PATH + 'vis/'
    os.makedirs(VIS_PATH, exist_ok=True)
    self.LOG_PATH = LOG_PATH
    self.VIS_PATH = VIS_PATH

  def gatherData(self, pop, species):
    """Collect and stores run data
    """

    # Readability
    fitness = [ind.fitness for ind in pop]
    nodes = np.asarray([np.shape(ind.node)[1] for ind in pop])
    conns = np.asarray([ind.nConn for ind in pop])
    
    # --- Evaluation Scale ---------------------------------------------------
    if len(self.x_scale) == 0:
      self.x_scale = np.append(self.x_scale, len(pop))
    else:
      self.x_scale = np.append(self.x_scale, self.x_scale[-1]+len(pop))
    # ------------------------------------------------------------------------ 

    
    # --- Best Individual ----------------------------------------------------
    self.elite.append(pop[np.argmax(fitness)])
    if len(self.best) == 0:
      self.best = copy.deepcopy(self.elite)
    elif (self.elite[-1].fitness > self.best[-1].fitness):
      self.best = np.append(self.best,copy.deepcopy(self.elite[-1]))
      self.newBest = True
    else:
      self.best = np.append(self.best,copy.deepcopy(self.best[-1]))   
      self.newBest = False
    # ------------------------------------------------------------------------ 

    
    # --- Generation fit/complexity stats ------------------------------------ 
    self.node_med = np.append(self.node_med,np.median(nodes))
    self.conn_med = np.append(self.conn_med,np.median(conns))
    self.fit_med  = np.append(self.fit_med, np.median(fitness))
    self.fit_max  = np.append(self.fit_max,  self.elite[-1].fitness)
    self.fit_top  = np.append(self.fit_top,  self.best[-1].fitness)
    # ------------------------------------------------------------------------ 


    # --- MOO Fronts ---------------------------------------------------------
    if self.p['alg_probMoo'] > 0:
      if len(self.objVals) == 0:
        self.objVals = np.c_[fitness,conns]
      else:
        self.objVals = np.c_[self.objVals, np.c_[fitness,conns]]
    # ------------------------------------------------------------------------ 

    
    # --- Species Stats ------------------------------------------------------
    if self.p['alg_speciate'] == 'neat':
      specFit = np.empty((2,0))
      #print('# of Species: ', len(species))
      for iSpec in range(len(species)):
        for ind in species[iSpec].members:
          tmp = np.array((iSpec,ind.fitness))
          specFit = np.c_[specFit,tmp]
      self.spec_fit = specFit
    # ------------------------------------------------------------------------ 


  def display(self):
    """Console output for each generation
    """
    return    "|---| Elite Fit: " + '{:.2f}'.format(self.fit_max[-1]) \
         + " \t|---| Best Fit:  "  + '{:.2f}'.format(self.fit_top[-1])

  def save(self, gen=(-1)):
    """Save algorithm stats to disk
    """
    ''' Save data to disk '''
    filename = self.filename
    pref = self.LOG_PATH + filename
    
    # --- Generation fit/complexity stats ------------------------------------ 
    # gStatLabel = ['x_scale',\
    #               'fit_med','fit_max','fit_top','node_med','conn_med']
    # genStats = np.empty((len(self.x_scale),0))
    # for i in range(len(gStatLabel)):
    #   #e.g.         self.    fit_max          [:,None]
    #   evalString = 'self.' + gStatLabel[i] + '[:,None]'
    #   genStats = np.hstack((genStats, eval(evalString)))
    # lsave(pref + '_stats.out', genStats)
    # ------------------------------------------------------------------------ 


    # --- Best Individual ----------------------------------------------------
    self.best[gen].save(pref + f'{gen}_best.json')
    from ..vis.viewInd import viewInd, fig2img 
    img = fig2img(viewInd(self.best[gen])[0])
    img.save(self.VIS_PATH + f'_{gen}_best.png')
    
    # ------------------------------------------------------------------------


    # --- Species Stats ------------------------------------------------------
    # if self.p['alg_speciate'] == 'neat':
    #   lsave(pref + '_spec.csv', self.spec_fit)
    # ------------------------------------------------------------------------


    # --- MOO Fronts ---------------------------------------------------------
    # if self.p['alg_probMoo'] > 0:
    #   lsave(pref + '_objVals.csv',self.objVals)
    # ------------------------------------------------------------------------

  def savePop(self,pop,filename):
    """Save all individuals in population as numpy arrays
    """
    folder = self.LOG_PATH + filename + '_pop/'
    if not os.path.exists(folder):
      os.makedirs(folder)

    for i in range(len(pop)):
      pop[i].save(folder+'ind_'+str(i)+'.json')

def lsave(filename, data):
  """Short hand for numpy save with csv and float precision defaults
  """
  np.savetxt(filename, data, delimiter=',',fmt='%1.2e')





