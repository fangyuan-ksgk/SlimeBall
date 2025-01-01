"""
State mode (Optional Human vs Built-in AI)

FPS (no-render): 100000 steps /7.956 seconds. 12.5K/s.
"""

import gym
import slimevolleygym
from slimevolleygym.mlp import makeSlimePolicy, makeSlimePolicyLite # simple pretrained models
from slimevolleygym import BaselinePolicy
from time import sleep


class RandomPolicy:
  def __init__(self, path):
    self.action_space = gym.spaces.MultiBinary(3)
    pass
  def predict(self, obs):
    return self.action_space.sample()

def makeBaselinePolicy(_):
  return BaselinePolicy()


MODEL = {
    "baseline": makeBaselinePolicy,
    "cma": makeSlimePolicy,
    "ga": makeSlimePolicyLite,
    "random": RandomPolicy,
  }


env = gym.make("SlimeVolley-v0")

# policy0 = MODEL["ga"]("zoo/ga_sp/ga.json")
policy0 = MODEL["baseline"]("")
policy1 = MODEL["baseline"]("")

obs0 = env.reset()
obs1 = obs0 # same observation at the very beginning for the other agent

done = False
total_reward = 0
render_mode = True


while not done:

    # agent absolute x,y,vx,xy | ball position x,y,vx,vy | opponent absolute x,y,vx,vy
    # Therefore, compared to agent1 observation, agent0 observation flipps last 4 with first 4 and negates ball's x & vx 
    action0 = policy0.predict(obs0)
    action1 = policy1.predict(obs1)

    # uses a 2nd (optional) parameter for step to put in the other action
    # and returns the other observation in the 4th optional "info" param in gym's step()
    obs0, reward, done, info = env.step(action0, action1)
    obs1 = info['otherObs']

    total_reward += reward

    if render_mode:
        env.render()
        """ # used to render stuff to a gif later.
        img = env.render("rgb_array")
        filename = os.path.join("gif","daytime",str(count).zfill(8)+".png")
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        count += 1
        """
        sleep(0.01)