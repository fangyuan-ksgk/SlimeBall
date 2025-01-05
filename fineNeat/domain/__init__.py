from .make_env import make_env
from .config import games
from .task_gym import *


load_task = lambda hyp: GymTask(games[hyp['task']],paramOnly=True)