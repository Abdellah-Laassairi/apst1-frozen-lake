
import gymnasium as gym
import pygame
import sys
import argparse
from tools.qlearning import *
from tools.sarsa import *
from tools.reinforce import *

def main(env_name, mode, learning_method, save, visualize):
    """
    main function to launch reinforcement learning agent
    """

    # Initialize visualization
    if visualize :
        print("Initializing visualization")
        render_mode = 'human'
    else :
        print("No visualization")
        render_mode = None

    # Environment selection
    if env_name == 'Acrobot':
        env = gym.make("Acrobot-v1", render_mode=render_mode)
        observation, info = env.reset(seed=42)
    elif env_name == 'Frozen_lake':
        env = gym.make("FrozenLake-v1", render_mode=render_mode)
        observation, info = env.reset(seed=42)


    if mode=="Train":
        # Learning method selection
        if learning_method == 'Qlearning':
            q_table = q_train_greedy_decay(env, alpha = 0.1, gamma = 0.99, min_epsilon=0.5)
            if save :
                np.save("data/"+env_name+"_"+learning_method, q_table)

        elif learning_method == 'SARSA':
            sarsa = sarsa_train(env, alpha = 0.1, gamma = 0.99, epsilon = 0.1, episodes = 10000, steps = 100)
            if save :
                np.save("data/"+env_name+"_"+learning_method, sarsa)
        
        elif learning_method == "REINFORCE":
            params = REINFORCE(env)
            if save :
                np.save("data/"+env_name+"_"+learning_method, params)
    
    elif mode=="Play" and learning_method == "REINFORCE":
        try :
            params = np.load("data/"+env_name+"_"+learning_method+".npy")
            rewards, num_steps = agent_play_r(env, params)
        except :
            print("Train the Agent using this method first")

    else : 
        try :
            table = np.load("data/"+env_name+"_"+learning_method+".npy")
            rewards, success_rate, avg_num_steps= agent_play(env, q_table=table)
        except :
            print("Train the Agent using this method first")


            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="Training",
        help=
        "options : Train, Play"
    )
    parser.add_argument(
        "--method",
        dest="method",
        type=str,
        default="Qlearning",
        help=
        "options : Qlearning, SARSA, REINFORCE"
    )
    parser.add_argument(
        "--env_name",
        dest="env_name",
        type=str,
        default="Frozen_lake",
        help="options : Frozen_lake, Acrobot")
    
    parser.add_argument(
        "--save",
        dest="save",
        type=bool,
        default=True,
        help="Save the trained model or weights")

    parser.add_argument(
        "--visualize",
        dest="visualize",
        type=bool,
        default=False,
        help="Visualize and render the environment")
    
    args = parser.parse_args()

    main(args.env_name, args.mode, args.method, args.save, args.visualize)

    """ 

    Frozen environment:

    Training example :
    python3 main.py --env_name Frozen_lake --mode Train --method Qlearning --save True

    Simulation example :
    python3 main.py --env_name Frozen_lake --mode Play --method Qlearning --visualize True 

    Acrobat environment:
    python3 main.py --env_name Acrobot --mode Train --method REINFORCE --save True
    python3 main.py --env_name Acrobot --mode Play --method REINFORCE --visualize True 

    """