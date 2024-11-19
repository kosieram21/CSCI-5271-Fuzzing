import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

import numpy as np

class Memory:
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.states = []
		self.actions = []
		self.action_log_probs = []
		self.rewards = []
		self.values = []
		self.is_terminals = []

	def store(self, state, action, action_log_prob, reward, value, is_terminal):
		self.states.append(state)
		self.actions.append(action)
		self.action_log_probs.append(action_log_prob)
		self.rewards.append(reward)
		self.values.append(value)
		self.is_terminals.append(is_terminal)

	def clear(self):
		self.states.clear()
		self.actions.clear()
		self.action_log_probs.clear()
		self.rewards.clear()
		self.values.clear()
		self.is_terminals.clear()

	def generate_batches(self):
		num_states = len(self.states)
		batch_starts = np.arange(0, num_states, self.batch_size)
		indicies = np.arange(num_states)
		np.random.shuffle(indicies)
		batches = [indicies[i:i+self.batch_size] for i in batch_starts]
		return \
			np.array(self.states), np.array(self.actions), np.array(self.action_log_probs), \
			np.array(self.rewards), np.array(self.values), np.array(self.is_terminals), batches
	
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim):
		super(Actor, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, action_dim),
			nn.Softmax(dim=-1))
		
	def forward(self, state):
		action_probs = self.mlp(state)
		action_distribution = Categorical(action_probs)
		return action_distribution
	
class Critic(nn.Module):
	def __init__(self, state_dim, hidden_dim):
		super(Critic, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, 1))
		
	def forward(self, state):
		value = self.mlp(state)
		return value
	
class Agent:
	def __init__(self, state_dim, action_dim, hidden_dim, batch_size, gamma, eps, lamda, lr, device):
		self.device = device
		
		self.gamma = gamma
		self.eps = eps
		self.lamda = lamda

		self.memory = Memory(batch_size)
		self.actor = Actor(state_dim, action_dim, hidden_dim)
		self.critic = Critic(state_dim, hidden_dim)

		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

	def choose_action(self, state):
		state = torch.tensor([state]).to(self.device)

		action_distribution = self.actor(state)
		action = action_distribution.sample()
		action_log_prob = torch.squeeze(action_distribution.log_prob(action)).item()
		action = torch.squeeze(action).item()

		value = self.critic(state)
		value = torch.squeeze(value).item()

		return action, action_log_prob, value
	
	def compute_advantages(self, rewards, values, is_terminals):
		time_steps = len(rewards)
		advantages = np.zeros(time_steps)

		for t in range(time_steps - 1):
			discount = 1
			advantage = 0
			
			for k in range(t, time_steps - 1):
				advantage += rewards[k] + (self.gamma * values[k+1]) * (1 - is_terminals[k]) - values[k]
				advantage *= discount
				discount *= self.gamma * self.lamda

			advantages[t] = advantage

		advantages = torch.tensor(advantages).to(self.device)
		return advantages
	
	def learn(self, epochs=1):
		for _ in range(epochs):
			states, actions, old_action_log_probs, rewards, values, is_terminals, batches = self.memory.generate_batches()
			advantages = self.compute_advantages(rewards, values, is_terminals)
			values = torch.tensor(values).to(self.device)

			for batch in batches:
				states = torch.tensor(states[batch]).to(self.device)
				actions = torch.tensor(actions[batch]).to(self.device)
				old_action_log_probs = torch.tensor(old_action_log_probs[batch]).to(self.device)

				new_action_distribution = self.actor(states)
				new_action_log_probs = new_action_distribution.log_prob(actions)
				action_prob_ratios = new_action_log_probs.exp() / old_action_log_probs.exp()
				weighted_action_probs = action_prob_ratios * advantages[batch]
				clipped_weighted_action_probs = torch.clamp(action_prob_ratios, 1 - self.eps, 1 + self.eps) * advantages[batch]
				actor_loss = -torch.min(weighted_action_probs, clipped_weighted_action_probs).mean()

				critic_values = self.critic(states)
				critic_values = torch.squeeze(critic_values)
				returns = advantages[batch] + values[batch]
				critic_loss = F.mse_loss(critic_values, returns)

				# 0.5 could be hyperparameter, we can also introduce entropy reward to encourage exploration
				total_loss = actor_loss + 0.5 * critic_loss
				self.actor_optimizer.zero_grad()
				self.critic_optimizer.zero_grad()
				total_loss.backward()
				self.actor_optimizer.step()
				self.critic_optimizer.step()

		self.memory.clear()
