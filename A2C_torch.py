'''
Advantage Actor-Critic algorithme (torch)
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
torch.manual_seed(0)

## A2C module
class A2C:
    ## initialization
    def __init__(self, env, gamma=0.99):
        self.gamma = gamma
        self.discount = 1.
        self.action_n = env.action_space.n
        self.actor_net = self.build_net(input_size=env.observation_space.shape[0], 
                                        hidden_sizes=[100,],
                                        output_size=env.action_space.n,
                                        output_activator=nn.Softmax(1))
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 0.001)
        self.critic_net = self.build_net(input_size=env.observation_space.shape[0], 
                                         hidden_sizes=[100,])
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), 0.001)
        self.critic_loss = nn.MSELoss()
    
    ## set A2C mode to None or train, default None
    def reset_mode(self, mode=None):
        self.mode = mode
        if self.mode=='train':
            self.traj = []
            self.discount = 1.
    
    ## take in the current observation and reward -> take an action 
    def play_step(self, observation, reward, done):
        state_tensor = torch.as_tensor(observation, dtype=torch.float).reshape(1, -1)
        proba_tensor = self.actor_net(state_tensor)
        action_tensor = dist.Categorical(proba_tensor).sample()
        action = action_tensor.numpy()[0]
        
        if self.mode=='train':
            self.traj += [observation, reward, done, action]
            if len(self.traj)>=8:
                self.reinforce()
            self.discount *= self.gamma
            
        return action
    
    ## build a neural network with ReLU activation function
    def build_net(self, input_size, hidden_sizes, output_size=1, 
                  output_activator=None):
        layers=[]
        
        # build layers
        for input_size, output_size in zip(
                [input_size,]+hidden_sizes, hidden_sizes+[output_size,]):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
        layers = layers[:-1]
        if output_activator:
            layers.append(output_activator)
            
        #build network
        net = nn.Sequential(*layers)
        
        return net
    
    # implementation of reinforce algorithme 
    def reinforce(self):
        state, _, _, action, next_state, reward, done, next_action = self.traj[-8:]
        state_tensor = torch.as_tensor(state, dtype=torch.float).unsqueeze(0)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float).unsqueeze(0)
        
        # TD error
        next_v_tensor = self.critic_net(next_state_tensor)
        target_tensor = reward + (1.-done)*self.gamma*next_v_tensor
        v_tensor = self.critic_net(state_tensor)
        td_error_tensor = target_tensor - v_tensor
        
        # train actor network
        # proba of taking current action at current state
        pi_tensor = self.actor_net(state_tensor)[0, action]
        logpi_tensor = torch.log(pi_tensor.clamp(1e-6, 1.))
        actor_loss_tensor = -(self.discount*td_error_tensor*logpi_tensor).squeeze()
        self.actor_optimizer.zero_grad()
        # gradients of the actor loss
        actor_loss_tensor.backward(retain_graph=True)
        # update the parameters of actor network using gradients
        self.actor_optimizer.step()
        
        # train critic network
        pred_tensor = self.critic_net(state_tensor)
        critic_loss_tensor = self.critic_loss(pred_tensor, target_tensor)
        self.critic_optimizer.zero_grad()
        # gradients of the critic loss
        critic_loss_tensor.backward()
        # update the parameters of critic network using gradients
        self.critic_optimizer.step()
        
## simulate an episode
## return the total reward of the episode and elapsed steps
def play_epis(env, agent, max_episode_steps=None, 
              mode=None, render=False):
    observation = env.reset()
    reward = 0.
    done = False
    agent.reset_mode(mode=mode)
    episode_reward = 0.
    elapsed_steps = 0
    
    while True:
        action = agent.play_step(observation, reward, done)
        if render:
            env.render()
        if done:
            break
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        elapsed_steps += 1
        if max_episode_steps and elapsed_steps>=max_episode_steps:
            break
            
        agent.close()
            
        return episode_reward, elapsed_steps