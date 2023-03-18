import numpy as np
import gymnasium as gym
from tqdm import *
import matplotlib.pyplot as plt


def softmax(alpha) :
    proba = np.exp(alpha)
    return proba/proba.sum()

def softmax_grad(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T) 

def parametrization(x,theta):
    z = x.dot(theta)
    return softmax(z)

def REINFORCE(env, learning_rate=0.000025, gamma=0.99, num_episodes=10000, max_steps=500):
    n_actions = env.action_space.n
    params = np.random.rand(6, n_actions)
    
    for _ in trange(num_episodes):
        state = env.reset()[0][None,:]
        grads = []
        rewards = []
        score = 0
        for _ in range(max_steps):
            probas = parametrization(state, params)
            action = np.random.choice(n_actions, p=probas[0])
            new_state, reward, done, truncated, _ = env.step(action)
            new_state = new_state[None,:] # For shape correctness
            dsoftmax = softmax_grad(probas)[action,:] # As stated before, we only keep the line of the current action
            dlog = dsoftmax / probas[0,action] # As expressed in the expression of the log-gradient, 
                                             # we divide by the value of the softmax at the current action
            grad = state.T.dot(dlog[None,:]) #Final dot product, shaped as a n x K matrix for shape correctness instead of
                                                # a nK vector
            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = new_state
            if done or truncated:
                break
        for i in range(len(grads)):
            params += learning_rate * grads[i] * sum([ r * (gamma ** r) for t,r in enumerate(rewards[i:])])
    return params

def agent_play_r(env,params, max_n_episodes=1000, max_steps=10000):
    count = 0
    rewards = []
    num_steps = []
    for _ in trange(max_n_episodes):
        s=env.reset()[0][None,:]
        total_reward = 0
        for i in range(max_steps):
            proba = parametrization(s, params)
            action = np.random.choice([0, 1, 2], p = proba[0])
            s, r, done, tr,_ = env.step(action)
            s = s[None,:]
            total_reward+=r
            if done or tr: 
                rewards.append(total_reward)
                num_steps.append(i+1)
                count+=1
                break
    avg_num_steps = np.mean(num_steps)
    print(f'Average number of steps = {avg_num_steps}')
    return rewards, num_steps