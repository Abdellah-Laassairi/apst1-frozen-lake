
import gymnasium as gym
import pygame
import sys

def main():
    """
    main function to launch reinforcement learning agent
    """
    env = gym.make("FrozenLake-v1", render_mode="human")
    observation, info = env.reset(seed=42)

    while True:
        keys = pygame.key.get_pressed()


        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
        env.reset()
        if keys[pygame.K_ESCAPE]:
            env.close()

            
if __name__ == '__main__':
    main()