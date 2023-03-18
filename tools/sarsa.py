import numpy as np
from tqdm import *

def action_choice_max(tab) :
    max_indices = np.argwhere(tab == np.amax(tab))
    max_indices = max_indices.reshape(len(max_indices),)
    action = np.random.choice(max_indices)
    return action


def sarsa_train(env, alpha=0.1, gamma=1, epsilon=0.5, episodes=10000, steps=1000) :
    "SARSA algorithme : les paramètres par défaut n'ont pas été ajustés."
    
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    q = np.zeros((num_states, num_actions))

    for ep in trange(episodes) : 
        
        env.reset()
        state = env.s

        if np.random.uniform() < epsilon :
            action = env.action_space.sample()
        else :
            action = action_choice_max(q[state, :])
            
        for _ in range(steps) :
            nw_state, reward, terminated, truncated, info = env.step(action)
            
            if np.random.uniform() < epsilon :
                nw_action = env.action_space.sample()
            else :
                nw_action = action_choice_max(q[nw_state, :])
                
            q[state, action] = (1-alpha)*q[state, action] + \
                alpha*(reward + gamma*q[nw_state, nw_action])
                
            state = nw_state
            action = nw_action
            
            if terminated or truncated :
                break
    
    env.close()
    
    return q