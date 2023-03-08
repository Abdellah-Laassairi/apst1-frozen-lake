import numpy as np
from tqdm import *

def sarsa_train(env, alpha, gamma, epsilon, q_init, episodes, steps) :
    
    q = q_init
    
    for ep in trange(episodes) : 
        
        env.reset()
        state = env.s
        
        if np.random.uniform() < epsilon :
            action = env.action_space.sample()
        else :
            action = np.argmax(q[state, :])
            
        for _ in range(steps) :
            nw_state, reward, terminated, truncated, info = env.step(action)
            
            if np.random.uniform() < epsilon :
                nw_action = env.action_space.sample()
            else :
                nw_action = np.argmax(q[nw_state, :])
                
            q[state, action] = (1-alpha)*q[state, action] + \
                alpha*(reward + gamma*q[nw_state, nw_action])
                
            state = nw_state
            action = nw_action
            
            if terminated or truncated :
                break
    
    env.close()
    
    return q