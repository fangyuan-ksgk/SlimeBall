"""
Human vs AI in pixel observation environment

Note that for multiagent mode, otherObs's image is horizontally flipped

Performance, 100,000 frames in 144.839 seconds, or 690 fps.
"""

import os
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"

import gym
from gym.envs.registration import register, make
import slimevolleygym
from slimevolleygym.slimevolley import (
    setPixelObsMode, WINDOW_WIDTH, WINDOW_HEIGHT,
    BALL_COLOR, AGENT_LEFT_COLOR, AGENT_RIGHT_COLOR,
    BACKGROUND_COLOR, GROUND_COLOR, FENCE_COLOR,
    REF_W, REF_H, REF_U, REF_WALL_WIDTH, REF_WALL_HEIGHT,
    FACTOR
)
import pygame
import numpy as np
from time import sleep

RENDER_MODE = True
SCALE = 4  # Scale factor for rendering

# Set pixel observation mode for rendering
setPixelObsMode()

if __name__=="__main__":

  manualAction = [0, 0, 0] # forward, backward, jump
  manualMode = False

  def handle_input():
    global manualMode, manualAction
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        print("Quit event received")
        return True
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
          manualAction[0] = 1
          print("Left pressed")
        if event.key == pygame.K_RIGHT:
          manualAction[1] = 1
          print("Right pressed")
        if event.key == pygame.K_UP:
          manualAction[2] = 1
          print("Jump pressed")
        if event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP]:
          manualMode = True
          print("Manual mode enabled")
        if event.key == pygame.K_q:
          print("Quit key pressed")
          return True
      elif event.type == pygame.KEYUP:
        if event.key == pygame.K_LEFT:
          manualAction[0] = 0
          print("Left released")
        if event.key == pygame.K_RIGHT:
          manualAction[1] = 0
          print("Right released")
        if event.key == pygame.K_UP:
          manualAction[2] = 0
          print("Jump released")
    return False

  def render_state(screen, state):
      """Render the game state using pygame"""
      screen.fill(BACKGROUND_COLOR)
      
      # Convert coordinates from game space to screen space
      bx, by = state[0]*WINDOW_WIDTH/REF_W + WINDOW_WIDTH/2, state[1]*FACTOR
      px, py = state[4]*WINDOW_WIDTH/REF_W + WINDOW_WIDTH/2, state[5]*FACTOR
      ox, oy = state[8]*WINDOW_WIDTH/REF_W + WINDOW_WIDTH/2, state[9]*FACTOR
      
      # Draw ground
      pygame.draw.rect(screen, GROUND_COLOR, (0, WINDOW_HEIGHT-REF_U*FACTOR, WINDOW_WIDTH, REF_U*FACTOR))
      
      # Draw net
      pygame.draw.rect(screen, FENCE_COLOR, 
                      (WINDOW_WIDTH/2-REF_WALL_WIDTH/2*FACTOR, 
                       WINDOW_HEIGHT-(REF_WALL_HEIGHT+REF_U)*FACTOR,
                       REF_WALL_WIDTH*FACTOR, 
                       REF_WALL_HEIGHT*FACTOR))
      
      # Draw ball
      pygame.draw.circle(screen, BALL_COLOR, (int(bx), WINDOW_HEIGHT-int(by)), 5*SCALE)
      
      # Draw left agent (player)
      pygame.draw.circle(screen, AGENT_LEFT_COLOR, (int(px), WINDOW_HEIGHT-int(py)), 15*SCALE)
      
      # Draw right agent (opponent)
      pygame.draw.circle(screen, AGENT_RIGHT_COLOR, (int(ox), WINDOW_HEIGHT-int(oy)), 15*SCALE)
      
      pygame.display.flip()

  # Initialize pygame and display if we're going to render
  if RENDER_MODE:
      print("\nInitializing pygame and display...")
      pygame.init()
      print("Pygame initialized successfully")
      print("Display driver:", pygame.display.get_driver())
      
      # Initialize display
      screen = pygame.display.set_mode((WINDOW_WIDTH*SCALE, WINDOW_HEIGHT*SCALE))
      pygame.display.set_caption("Slime Volley - Pixel Mode")
      print(f"Window created: {WINDOW_WIDTH*SCALE}x{WINDOW_HEIGHT*SCALE}")

  # Create environment without wrappers
  print("\nCreating environment...")
  env = make("SlimeVolley-v0", disable_env_checker=True)
  env.seed(np.random.randint(0, 10000))
  print("Environment created successfully")

  # Get initial observation
  print("\nResetting environment...")
  obs = env.reset()
  state = obs  # Initial state is the same as observation
  print("Initial observation shape:", obs.shape)
  
  # Initialize actions as numpy arrays
  manualAction = np.zeros(3, dtype=np.bool_)
  defaultAction = np.zeros(3, dtype=np.bool_)
  
  print("\nGame Controls:")
  print("Left Arrow  - Move Left")
  print("Right Arrow - Move Right")
  print("Up Arrow    - Jump")
  print("M          - Toggle Manual/AI Mode")
  print("Q          - Quit")

  policy = slimevolleygym.BaselinePolicy() # throw in a default policy (based on state, not pixels)

  defaultAction = np.array([0, 0, 0])  # Use numpy array like test_state.py
  running = True
  frame_count = 0  # Keep track of frames for debugging
  while running:
    # Handle pygame events and check if we should quit
    running = not handle_input()
    
    if not manualMode:
      action = policy.predict(obs)
    else:
      action = manualAction

    obs, reward, done, info = env.step(action)
    state = info['state']  # Get the actual state for rendering
    defaultAction = policy.predict(state)
    
    if RENDER_MODE:
        handle_input()
        # Render the game state using pygame
        render_state(screen, state)
        
        # Print debug info every 30 frames
        if frame_count % 30 == 0:
            print(f"\nFrame {frame_count}:")
            print(f"Action: {action}")
            print(f"Reward: {reward}")
            print("-" * 40)
        frame_count += 1
        sleep(0.02)  # Control frame rate
    
    if done:
      obs = env.reset()
    
    sleep(0.02)

  pygame.quit()
  env.close()
