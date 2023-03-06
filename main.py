
import gymnasium as gym
import pygame
import sys
import argparse


def main(env, method):
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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        dest="method",
        type=str,
        default="q-learning",
        help=
        "options : q-learning, SARSA"
    )
    parser.add_argument(
        "--env",
        dest="env",
        type=str,
        default="frozen_lake",
        help="options : frozen_lake, acrobot")
    
    args = parser.parse_args()


    main()