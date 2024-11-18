import math
import string
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

class Mlp(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
		super(Mlp, self).__init__()
		dim_list = [input_dim] + [hidden_dim] * num_layers + [output_dim]
		layers = []
		for i in range(len(dim_list) - 1):
			layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
			if i < len(dim_list) - 2:
				layers.append(nn.LayerNorm(dim_list[i + 1]))
				layers.append(nn.ReLU())
		self.net = nn.Sequential(*layers)

	def forward(self, x):
		return self.net(x)

class Actor(nn.Module): # policy pi
	def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim, encoder_layers, decoder_layers):
		super(Actor, self).__init__()
		self.gru = nn.GRU(input_dim, embedding_dim, encoder_layers, batch_first=True)
		self.action_decoder = Mlp(embedding_dim, output_dim, hidden_dim, decoder_layers)
		self.arg1_head = Mlp(output_dim + embedding_dim, 1, hidden_dim, decoder_layers)
		self.arg2_head = Mlp(output_dim + embedding_dim, 1, hidden_dim, decoder_layers)

	def forward(self, state):
		embedding, _ = self.gru(state)
		embedding = embedding[:, -1, :]
		action_embedding = self.action_decoder(embedding)
		joint_embedding = torch.cat((action_embedding, embedding), dim = 1)
		arg1_embedding = self.arg1_head(joint_embedding)
		arg2_embedding = self.arg2_head(joint_embedding)
		action_distribution = torch.softmax(action_embedding, dim=1)
		arg1 = torch.sigmoid(arg1_embedding)
		arg2 = torch.sigmoid(arg2_embedding)
		return action_distribution, arg1, arg2
	
class Critic(nn.Module): # Q function
	def __init__(self, input_dim, embedding_dim, hidden_dim, encoder_layers, decoder_layers):
		super(Critic, self).__init__()
		self.gru = nn.GRU(input_dim, embedding_dim, encoder_layers, batch_first=True)
		self.mlp = Mlp(embedding_dim + 3, 1, hidden_dim, decoder_layers)

	def forward(self, state, action, arg1, arg2):
		embedding, _ = self.gru(state)
		embedding = embedding[:, -1, :]
		input = torch.cat((embedding, action, arg1, arg2), dim=-1)
		reward = self.mlp(input)
		return reward
	
class TrainingPipeline:
	def __init__(self, actor, critic, actor_lr=0.001, critic_lr=0.001, gamma=0.99):
		self.memory = []
		self.actor = actor
		self.critic = critic
		self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
		self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
		self.criterion = nn.MSELoss()
		self.gamma = gamma

	def record_experience(self, state, next_state, action, arg1, arg2, reward):
		self.memory.append((state, next_state, action, arg1, arg2, reward))

	def replay_experiences(self):
		if not self.memory:
			return

		for state, next_state, action, arg1, arg2, reward in self.memory:
			with torch.no_grad():
				next_action_distribution, next_arg1, next_arg2 = self.actor(next_state)
				next_action = torch.multinomial(next_action_distribution, num_samples=1)
				next_value = self.critic(next_state, next_action, next_arg1, next_arg2)
				target = reward + self.gamma * next_value

			estimated_value = self.critic(state, action, arg1, arg2)
			critic_loss = self.criterion(estimated_value, target)
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			action_distribution, arg1, arg2 = self.actor(state)
			action = torch.multinomial(action_distribution, num_samples=1)
			actor_loss = -self.critic(state, action, arg1, arg2)
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

		self.memory.clear()

class ToyEnvironment:
	def __init__(self, initial_state, epsilon, epsilon_decay):
		self.initial_state = initial_state
		self.state = initial_state
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay

	def _lerp(self, input, start, end):
		return math.floor(((end - start) * input) + start)

	def _character_change(self, arg1, arg2):
		character_pos = self._lerp(arg1, 0, len(self.state) - 1)
		new_character = chr(self._lerp(arg2, 65, 122))
		# jank city
		state_list = list(self.state)
		state_list[character_pos] = new_character
		self.state = ''.join(state_list)

	def _transpose(self, arg1, arg2):
		pos1 = self._lerp(arg1, 0, len(self.state) - 1)
		pos2 = self._lerp(arg2, 0, len(self.state) - 1)
		# jank city
		state_list = list(self.state)
		state_list[pos1], state_list[pos2] = state_list[pos2], state_list[pos1]
		self.state = ''.join(state_list)

	def _insert_before(self, arg1, arg2):
		character_pos = self._lerp(arg1, 0, len(self.state) - 1)
		new_character = chr(self._lerp(arg2, 65, 122))
		self.state = self.state[:character_pos] + new_character + self.state[character_pos:]

	def _insert_after(self, arg1, arg2):
		character_pos = self._lerp(arg1, 0, len(self.state) - 1)
		new_character = chr(self._lerp(arg2, 65, 122))
		self.state = self.state[:character_pos+1] + new_character + self.state[character_pos+1:]

	def _delete(self, arg1):
		if len(self.state) <= 1:
			return
		character_pos = self._lerp(arg1, 0, len(self.state) - 1)
		self.state = self.state[:character_pos] + self.state[character_pos+1:]

	def _apply_mutation(self, chosen_action, arg1, arg2):
		if chosen_action == 0:
			self._character_change(arg1, arg2)
		elif chosen_action == 1:
			self._transpose(arg1, arg2)
		elif chosen_action == 2:
			self._insert_before(arg1, arg2)
		elif chosen_action == 3:
			self._insert_after(arg1, arg2)
		elif chosen_action == 4:
			self._delete(arg1)
		else:
			print('we should not be here')

	def _evaluate_state(self):
		reward = 0
		seen = set()
		for letter in string.ascii_letters:
			if letter in self.state and letter not in seen:
				reward += 5
				seen.add(letter)
		# if len(self.state) > 0 and self.state[0] == '[':
		# 	reward += 1000
		# reward -= len(self.state)
		return reward

	def reset(self):
		self.state = self.initial_state
		self.epsilon *= self.epsilon_decay
		return self.state
	
	def step(self, action, arg1, arg2):
		chosen_action = action if random.random() < (1 - self.epsilon) else random.randint(0, 4)
		arg1 = arg1 if random.random() < (1 - self.epsilon) else random.uniform(0, 1)
		arg2 = arg2 if random.random() < (1 - self.epsilon) else random.uniform(0, 1)
		initial_reward = self._evaluate_state()
		self._apply_mutation(chosen_action, arg1, arg2)
		final_reward = self._evaluate_state()
		return chosen_action, arg1, arg2, self.state, final_reward - initial_reward
	
def run_env(env, policy, device, training_pipeline, train=False, action_horizion=25):
	next_state = env.reset()
	total_reward = 0

	for _ in range(action_horizion):
		state_embedding = torch.tensor([float(ord(c)) for c in next_state]).unsqueeze(0).unsqueeze(-1).to(device)
		with torch.no_grad():
			action_distribution, arg1, arg2 = policy(state_embedding)
			action = torch.multinomial(action_distribution, num_samples=1)
		action, arg1, arg2, next_state, reward = env.step(action.item(), arg1.item(), arg2.item())
		next_state_embedding = torch.tensor([float(ord(c)) for c in next_state]).unsqueeze(0).unsqueeze(-1).to(device)
		total_reward += reward
		reward = torch.tensor(reward).to(device)
		action = torch.tensor(action).unsqueeze(0).unsqueeze(-1).to(device)
		arg1 = torch.tensor(arg1).unsqueeze(0).unsqueeze(-1).to(device)
		arg2 = torch.tensor(arg2).unsqueeze(0).unsqueeze(-1).to(device)
		if train:
			training_pipeline.record_experience(state_embedding, next_state_embedding, action, arg1, arg2, reward)

	if train:
		training_pipeline.replay_experiences()

	return env.state, total_reward


device = 'cuda' if torch.cuda.is_available() else 'cpu'

actor = Actor(
	input_dim=1,
	output_dim=5,
	embedding_dim=16,
	hidden_dim=32,
	encoder_layers=4,
	decoder_layers=4).to(device)

critic = Critic(
	input_dim=1,
	embedding_dim=16,
	hidden_dim=32,
	encoder_layers=4,
	decoder_layers=4).to(device)

training_pipeline = TrainingPipeline(actor, critic)

environment = ToyEnvironment('seed', 1, 0.99)
reward_lst = []

num_steps = 125
num_episodes = 500

for i in range(num_episodes):
	final_state, total_reward = run_env(environment, actor, device, training_pipeline, True, num_steps)
	print(f'Episode: {i}, Final State: {final_state}, Total Reward: {total_reward}, Code Coverage: {environment._evaluate_state()}')
	reward_lst.append(total_reward)

final_state, total_reward = run_env(environment, actor, device, training_pipeline, False, num_steps)
print(f'Final State: {final_state}, Total Reward: {total_reward}')

plt.plot(reward_lst, marker='o')
plt.title("Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()