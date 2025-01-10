"""
Simple MLP policy trained via estool (saved in /zoo)

code from https://github.com/hardmaru/estool
"""

import numpy as np
import json
from collections import namedtuple
# import neat 
import pickle 
from typing import Union

Game = namedtuple('Game', ['env_name', 'time_factor', 'input_size', 'output_size', 'layers', 'activation', 'noise_bias', 'output_noise', 'rnn_mode'])

games = {}

games['slimevolley'] = Game(env_name='SlimeVolley',
  input_size=12,
  output_size=3,
  time_factor=0,
  layers=[20, 20], # hidden size of 20x20 neurons
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
)

games['slimevolleylite'] = Game(env_name='SlimeVolley',
  input_size=12,
  output_size=3,
  time_factor=0,
  layers=[10, 10], # hidden size of 20x20 neurons
  activation='tanh',
  noise_bias=0.0,
  output_noise=[False, False, False],
  rnn_mode=False,
)

def makeSlimePolicy(filename):
  model = Model(games['slimevolley']) 
  model.load_model(filename)
  return model

def makeSlimePolicyLite(filename):
  model = Model(games['slimevolleylite']) 
  model.load_model(filename)
  return model

from typing import Optional 

class Model:
  ''' simple feedforward model '''
  def __init__(self, game, hidden_layers: Optional[list] = None):
    self.output_noise = game.output_noise # for volleyball, no output noise
    self.env_name = game.env_name
    # extend to multiple layers 
    if hidden_layers is None: 
      layers = game.layers
    else:
      layers = hidden_layers 
    
    self.rnn_mode = False # in the future will be useful
    self.time_input = 0 # use extra sinusoid input
    self.sigma_bias = game.noise_bias # bias in stdev of output
    self.sigma_factor = 0.5 # multiplicative in stdev of output
    if game.time_factor > 0:
      self.time_factor = float(game.time_factor)
      self.time_input = 1
    self.input_size = game.input_size
    self.output_size = game.output_size
    
    # make shapes from layers 
    self.shapes = []
    for i in range(len(layers)): 
      assert layers[i] > 0, "layer size must be positive"
      if layers[i] == 0: 
        break
      if i == 0: 
        self.shapes.append((self.input_size + self.time_input, layers[i]))
      else: 
        self.shapes.append((layers[i-1], layers[i]))
        
    self.shapes.append((layers[-1], self.output_size))

    self.sample_output = False
    l = len(self.shapes)
    if game.activation == 'relu':
      self.activations = [relu] * (l-1) + [passthru]
    elif game.activation == 'sigmoid':
      self.activations = [np.tanh] * (l-1) + [sigmoid]
    elif game.activation == 'softmax':
      self.activations = [np.tanh] * (l-1) + [softmax]
      self.sample_output = True
    elif game.activation == 'passthru':
      self.activations = [np.tanh] * (l-1) + [passthru]
    else:
      self.activations = [np.tanh] * l

    self.weight = []
    self.bias = []
    self.bias_log_std = []
    self.bias_std = []
    self.param_count = 0

    idx = 0
    for shape in self.shapes:
      self.weight.append(np.zeros(shape=shape))
      self.bias.append(np.zeros(shape=shape[1]))
      self.param_count += (np.product(shape) + shape[1])
      if self.output_noise[min(idx, len(self.output_noise)-1)]: # makeshift
        self.param_count += shape[1]
      log_std = np.zeros(shape=shape[1])
      self.bias_log_std.append(log_std)
      out_std = np.exp(self.sigma_factor*log_std + self.sigma_bias)
      self.bias_std.append(out_std)
      idx += 1

    self.render_mode = False

  def make_env(self, seed=-1, render_mode=False):
    self.render_mode = render_mode
    self.env = make_env(self.env_name, seed=seed, render_mode=render_mode)

  def predict(self, x, t=0, mean_mode=False):
    # if mean_mode = True, ignore sampling.
    h = np.array(x).flatten()
    if self.time_input == 1:
      time_signal = float(t) / self.time_factor
      h = np.concatenate([h, [time_signal]])
    num_layers = len(self.weight)
    for i in range(num_layers):
      w = self.weight[i]
      b = self.bias[i]
      h = np.matmul(h, w) + b
      h = self.activations[i](h)

    if self.sample_output:
      h = sample(h)

    return h
  
  def get_model_params(self):
    """ 
    Returns flattened array of all model parameters (weights and biases)
    """
    params = []
    for i in range(len(self.shapes)):
        # Flatten and append weights
        params.extend(self.weight[i].flatten())
        # Append biases
        params.extend(self.bias[i].flatten())
        # If output noise is enabled for this layer, append log_std parameters
        if self.output_noise[i]:
            params.extend(self.bias_log_std[i].flatten())
    return np.array(params)

  def set_model_params(self, model_params):
    """ 
    FY: "Unflatten" model_params into weight and bias via reshape
    """
    pointer = 0
    for i in range(len(self.shapes)):
      w_shape = self.shapes[i]
      b_shape = self.shapes[i][1]
      s_w = np.product(w_shape)
      s = s_w + b_shape
      chunk = np.array(model_params[pointer:pointer+s])
      self.weight[i] = chunk[:s_w].reshape(w_shape)
      self.bias[i] = chunk[s_w:].reshape(b_shape)
      pointer += s
      if self.output_noise[i]:
        s = b_shape
        self.bias_log_std[i] = np.array(model_params[pointer:pointer+s])
        self.bias_std[i] = np.exp(self.sigma_factor*self.bias_log_std[i] + self.sigma_bias)
        if self.render_mode:
          print("bias_std, layer", i, self.bias_std[i])
        pointer += s

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    return np.random.randn(self.param_count)*stdev
  
  @classmethod
  def from_indiv(cls, indiv, game): 
      params = indiv.to_params()
      hidden_shapes = [p[1].shape[0] for p in params][:-1]
      new_policy = Model(game, hidden_layers=hidden_shapes)
      output_noise = np.zeros(shape=len(params), dtype=bool)
      for i in range(len(params)): 
          new_policy.weight[i] = params[i][0]
          new_policy.bias[i] = params[i][1]
      new_policy.output_noise = output_noise
      return new_policy



