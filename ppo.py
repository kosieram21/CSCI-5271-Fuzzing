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

	# todo: this should be able to be simplified. if we could reproduce the same functionality
	# in torch instead of numpy we would not have to swap between torch.tensors and np.arrays
	# greatly simplifying the code
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

	# we can easily adapt this to model a continous action space simply by changing the
	# distribution (which shouldn't impact the rest of the code). instead of having the model
	# output the logits for a categorical (discrete pmf) distribution we can instead have the
	# model output the mean of a normal distribution. then when we sample we will be taking actions
	# near the mean. the amount of exploration we do will depend on the standard deviation. this will
	# not be output by the model but instead will be an external hyperparameter that we can anneal
	# over time to balance exploration and explotation	
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
	
	# generalized advantage estimate (gae) computation. 
	# naive advantage A(t) = R(t) + V(s(t+1)) - V(s(t)) 
	# high variance since V (the critic) only approximates the true value
	# monte carlo advantage A(t) = G(t) - V(s(t)) where G(t) is the discounted observed reward
	# high bias do to incomplete modeling of future rewards. really A(t) = Q(s(t), a(t)) - V(s(t))
	# but monte carlo advantage does not model Q it simply relies on G and therefore ignores all
	# of the other possible trajectories except for the observed trajectoru introducing bias.
	# gae aims to strike a balance between naive advantages and monte carlo advantages and thus
	# strikes a balance between variance and bias stabalizing training. not if we choose lambda
	# to be 0 gae collapses to monte carlo advantages.
	def compute_generalized_advantage_estimates(self, rewards, values, is_terminals):
		time_steps = len(rewards)
		advantages = np.zeros(time_steps)
		last_advantage = 0

		for t in reversed(range(time_steps)):
			if t == time_steps - 1:
				delta = rewards[t] - values[t]
			else:
				delta = rewards[t] + self.gamma * values[t + 1] * (1 - is_terminals[t]) - values[t]

			advantages[t] = delta + self.gamma * self.lamda * last_advantage * (1 - is_terminals[t])
			last_advantage = advantages[t]

		advantages = torch.tensor(advantages).to(self.device)
		return advantages

	# standard (monte carlo) advantage computation. the advantage is the differences between the actual return
	# of the policy given the state and the estimated value of the state. positive advantage means
	# the policy is performing better than expected while negative advantage indicates the policy
	# is performing worse than expeected.
	def compute_advantages(self, rewards, values, is_terminals):
		time_steps = len(rewards)
	
		returns = np.zeros(time_steps)
		cumulative_reward = 0
		for t in reversed(range(time_steps)):
			if is_terminals[t]:
				cumulative_reward = 0
			cumulative_reward = rewards[t] + self.gamma * cumulative_reward
			returns[t] = cumulative_reward
	
		advantages = returns - values
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

				# actor_loss is the clipped surragate objective. if the new policy has an advantage over 
				# the value baseline this objective increases the probability of that action otherwise
				# it will decrease the probability. this objective also utilizes clipping to prevent the
				# policy from rapidly updating away from the current policy. this is because large changes
				# to the model weights can easily destabalize the policy in reinforcement learning.
				# critic loss is the mean squared error betwwen the actual returns and the value estimate.
				# this essentially regresses the critic on the bellman equation providing a mechanism for
				# value bootstraping and increasing the accuracy of the advantage calculation.
				# the final term is the policy entropy. by encouraging high entropy in the policy we encourage
				# exploration of the state space.
				total_loss = actor_loss + 0.5 * critic_loss - 0.1 * new_action_distribution.entropy()
				self.actor_optimizer.zero_grad()
				self.critic_optimizer.zero_grad()
				total_loss.backward()
				self.actor_optimizer.step()
				self.critic_optimizer.step()

		self.memory.clear()