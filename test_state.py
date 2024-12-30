"""
State mode (Optional Human vs Built-in AI)

FPS (no-render): 100000 steps /7.956 seconds. 12.5K/s.
"""

import math
import numpy as np
import gym
import slimevolleygym

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

# game settings:

RENDER_MODE = True


if __name__=="__main__":
  """
  Example of how to use Gym env, in single or multiplayer setting

  Humans can override controls:

  blue Agent:
  W - Jump
  A - Left
  D - Right

  Yellow Agent:
  Up Arrow, Left Arrow, Right Arrow
  """

  if RENDER_MODE:
    import pygame
    from time import sleep
    pygame.init()  # Initialize pygame

  manualAction = [0, 0, 0] # forward, backward, jump
  otherManualAction = [0, 0, 0]
  manualMode = False
  otherManualMode = False

  def handle_input():
    global manualMode, manualAction, otherManualMode, otherManualAction
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:  manualAction[0] = 1
            if event.key == pygame.K_RIGHT: manualAction[1] = 1
            if event.key == pygame.K_UP:    manualAction[2] = 1
            if event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP]: 
                manualMode = True

            if event.key == pygame.K_d:     otherManualAction[0] = 1
            if event.key == pygame.K_a:     otherManualAction[1] = 1
            if event.key == pygame.K_w:     otherManualAction[2] = 1
            if event.key in [pygame.K_d, pygame.K_a, pygame.K_w]: 
                otherManualMode = True

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:  manualAction[0] = 0
            if event.key == pygame.K_RIGHT: manualAction[1] = 0
            if event.key == pygame.K_UP:    manualAction[2] = 0
            if event.key == pygame.K_d:     otherManualAction[0] = 0
            if event.key == pygame.K_a:     otherManualAction[1] = 0
            if event.key == pygame.K_w:     otherManualAction[2] = 0

  policy = slimevolleygym.BaselinePolicy() # defaults to use RNN Baseline for player
  otherPolicy = slimevolleygym.BaselinePolicy() # TODO: change to another policy (?)

  env = gym.make("SlimeVolley-v0")
  env.seed(np.random.randint(0, 10000))
  #env.seed(689)

  if RENDER_MODE:
    env.render()

  obs = env.reset()
  otherObs = obs  # Initialize otherObs with the same initial observation

  steps = 0
  total_reward = 0
  action = np.array([0, 0, 0])

  done = False

  while True:
    if not manualMode:
        action = policy.predict(obs)
    else:
        action = manualAction

    if not otherManualMode:
        otherAction = otherPolicy.predict(otherObs)
    else:
        otherAction = otherManualAction

    obs, reward, done, info = env.step(action, otherAction)
    otherObs = info['otherObs']

    if RENDER_MODE:
        handle_input()
        env.render()
        sleep(0.02)

    if done:
        obs = env.reset()
        otherObs = obs # same observation at the beginning

  env.close()
  print("cumulative score", total_reward)
