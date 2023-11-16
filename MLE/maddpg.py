"""
Title: Multi-Agent Deep Deterministic Policy Gradient (MADDPG)



"""

from copy import deepcopy
from typing import List
import os
import logging

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

def setup_logger(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
    

class Agent:
    """ A single agent in MADDPG """
    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr, device=None):
        """
        An agent in MADDPG
        :param obs_dim: observation dimension of the current agent, i.e. local observation space
        :param act_dim: action dimension of the current agent, i.e. local action space
        :param global_obs_dim: input dimension of the global critic of the current agent, if there are
        :param actor_lr: learning rate of the actor
        :param critic_lr: learning rate of the critic
        :param device: torch.device
        3 agents for example, the input for global critic is (obs1, obs2, obs3, act1, act2, act3)
        """
        
        # Actor output logit of each action
        self.actor = MLPNetwork(obs_dim, act_dim)
        # critic input all the states and actions
        self.critic = MLPNetwork(global_obs_dim, 1)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.device = device

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return torch.nn.functional.softmax(logits / tau, dim=-1)
    
    def action(self, obs, model_out=False):
        """
        Choose action according to given `obs`
        :param obs: observation
        :param model_out: the original output of the actor, i.e. the logits of each action will be
                        : `gumbel_softmax`ed by default(model_out=False) and only the action will be returned
                        : if set to True, return the original output of the actor and the action
        """
        
        logits = self.actor(obs)
        action = self.gumbel_softmax(logits)
        if model_out:
            return action, logits
        return action

    def target_action(self, obs):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])

        logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        action = self.gumbel_softmax(logits)
        return action.squeeze(0).detach()
    
    def critic_value(self, state_list, act_list):
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length

    def target_critic_value(self, state_list, act_list):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)
    
    
class Buffer:
    def __init__(self, capacity, obs_dim, act_dim):
        """
        Multi-agent replay buffer
        
        :param capacity: capacity of the replay buffer
        :param obs_dim: observation dimension of the current agent, i.e. local observation space
        :param act_dim: action dimension of the current agent, i.e. local action space
        """
        self.capacity = capacity
    
        self.obs = np.zeros((capacity, obs_dim))
        self.action = np.zeros((capacity, act_dim))
        self.reward = np.zeros(capacity)
        self.next_obs = np.zeros((capacity, obs_dim))
        self.done = np.zeros(capacity, dtype=bool)
        
        self._index = 0
        self._size = 0
        
        self.device = "cpu"
        
    def add(self, obs, action, reward, next_obs, done):
        """ add an experience to the memory """
        self.obs[self._index] = obs
        self.action[self._index] = action
        self.reward[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.done[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1
        
    def sample(self, indices):
        """ """
        obs = self.obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_obs = self.next_obs[indices]
        done = self.done[indices]

        # NOTE that `obs`, `action`, `next_obs` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        obs = torch.from_numpy(obs).float().to(self.device)  # torch.Size([batch_size, state_dim])
        action = torch.from_numpy(action).float().to(self.device)  # torch.Size([batch_size, action_dim])
        reward = torch.from_numpy(reward).float().to(self.device)  # just a tensor with length: batch_size
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)  # Size([batch_size, state_dim])
        done = torch.from_numpy(done).float().to(self.device)  # just a tensor with length: batch_size

        return obs, action, reward, next_obs, done

    def __len__(self):
        return self._size

    
    
class MADDPG:
    def __init__(self, obs_dim_list, act_dim_list, capacity, actor_lr, critic_lr, res_dir, device=None):
        """
        
        :param obs_dim_list: list of observation dimension of each agent
        :param act_dim_list: list of action dimension of each agent
        :param capacity of the replay buffer
        :param actor_lr: learning rate of the actor
        :param critic_lr: learning rate of the critic
        :param res_dir: directory where log file and all the data and figures will be saved
        :param device: torch.device     
        """
        
        # TODO Device
        self.device = "cpu"
        
        # Sum all of the dimensions of each agent to get input dimension for critic
        global_obs_dim = sum(obs_dim_list + act_dim_list)
        
        # Create all the agents and corresponding replay buffer
        self.agents: List[Agent] = []
        self.buffers: List[Buffer] = []
        for obs_dim, act_dim in zip(obs_dim_list, act_dim_list):
            self.agents.append(Agent(obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr))
            self.buffers.append(Buffer(capacity, obs_dim, act_dim))
            
        # Handle output
        if res_dir is None:
            self.logger = setup_logger('maddpg.log')
        else:
            self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log'))
            
    def add_experience(self, obs, actions, rewards, next_obs, dones):
        """ Add experience to buffer"""
        for n, buffer in enumerate(self.buffers):
            # print(f"{n}: obs: {obs[n]}, actions: {actions[n]}, rewards: {rewards[n]}, next_obs: {next_obs[n]}, dones: {dones[n]}")
            buffer.add(obs[n], actions[n], rewards[n], next_obs[n], dones[n])
            
    def sample_experience(self, batch_size, agent_index):
        """ Sample experience from all the agents' buffers, and collect data for network input"""
        # Get the total number of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers[0])
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        obs_list, act_list, next_obs_list, next_act_list = [], [], [], []
        reward_cur, done_cur, obs_cur = None, None, None
        for n, buffer in enumerate(self.buffers):
            obs, action, reward, next_obs, done = buffer.sample(indices)
            obs_list.append(obs)
            act_list.append(action)
            next_obs_list.append(next_obs)
            # calculate next_action using target_network and next_state
            next_act_list.append(self.agents[n].target_action(next_obs))
            if n == agent_index:  # reward and done of the current agent
                obs_cur = obs
                reward_cur = reward
                done_cur = done

        return obs_list, act_list, reward_cur, next_obs_list, done_cur, next_act_list, obs_cur
        
    def select_action(self, obs):
        """ """
        actions = []
        for n, agent in enumerate(self.agents):  # each agent select action according to their obs
            o = torch.from_numpy(obs[n]).unsqueeze(0).float().to(self.device)  # torch.Size([1, state_size])
            # Note that the result is tensor, convert it to ndarray before input to the environment
            act = agent.action(o).squeeze(0).detach().cpu().numpy()
            actions.append(act)
            # self.logger.info(f'agent {n}, obs: {obs[n]} action: {act}')
        return actions
        
    
    def learn(self, batch_size, gamma):
        """ """
        for i, agent in enumerate(self.agents):
            obs, act, reward_cur, next_obs, done_cur, next_act, obs_cur = self.sample_experience(batch_size, i)
            # update critic
            critic_value = agent.critic_value(obs, act)

            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(next_obs, next_act)
            target_value = reward_cur + gamma * next_target_critic_value * (1 - done_cur)

            critic_loss = torch.nn.functional.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)

            # update actor
            # action of the current agent is calculated using its actor
            action, logits = agent.action(obs_cur, model_out=True)
            act[i] = action
            actor_loss = -agent.critic_value(obs, act).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
            # self.logger.info(f'agent{i}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')
            
    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents:
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)
