import math
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

class Actor(nn.Module): # policy pi
	def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim, encoder_layers, decoder_layers):
		super(Actor, self).__init__()
		self.gru = nn.GRU(input_dim, embedding_dim, encoder_layers, batch_first=True)
		dim_list = [embedding_dim] + [hidden_dim] * decoder_layers + [output_dim]
		layers = []
		for i in range(len(dim_list) - 1):
			layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
			if i < len(dim_list) - 2:
				layers.append(nn.LayerNorm(dim_list[i + 1]))
				layers.append(nn.ReLU())
		self.mlp = nn.Sequential(*layers)

	def forward(self, state):
		embedding, _ = self.gru(state)
		embedding = embedding[:, -1, :]
		action = self.mlp(embedding)
		return action
	
class Critic(nn.Module): # Q function
	def __init__(self, state_input_dim, action_input_dim, output_dim, embedding_dim, hidden_dim, encoder_layers, decoder_layers):
		super(Critic, self).__init__()
		self.gru = nn.GRU(state_input_dim, embedding_dim, encoder_layers, batch_first=True)
		dim_list = [embedding_dim + action_input_dim] + [hidden_dim] * decoder_layers + [output_dim]
		layers = []
		for i in range(len(dim_list) - 1):
			layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
			if i < len(dim_list) - 2:
				layers.append(nn.LayerNorm(dim_list[i + 1]))
				layers.append(nn.ReLU())
		self.mlp = nn.Sequential(*layers)

	def forward(self, state, action):
		embedding, _ = self.gru(state)
		embedding = embedding[:, -1, :]
		input = torch.cat((embedding, action), dim=-1)
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

	def record_experience(self, state, next_state, action, reward):
		self.memory.append((state, next_state, action, reward))

	def replay_experiences(self):
		if not self.memory:
			return

		for state, next_state, action, reward in self.memory:
			with torch.no_grad():
				next_action = self.actor(next_state)
				next_value = self.critic(next_state, next_action)
				target = reward + self.gamma * next_value

			estimated_value = self.critic(state, action)
			critic_loss = self.criterion(estimated_value, target)
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			predicted_action = self.actor(state)
			#print(f'predicted action: {predicted_action}')
			actor_loss = -self.critic(state, predicted_action)
			#print(f' actor loss: {actor_loss}')
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

		self.memory.clear()

class ToyEnvironment:
	def __init__(self, initial_state, temperature, epsilon):
		self.initial_state = initial_state
		self.state = initial_state
		self.temperature = temperature
		self.epsilon = epsilon

	def _lerp(self, input, start, end):
		return math.floor(((end - start) * input) + start)

	def _character_change(self, arg1, arg2):
		character_pos = self._lerp(arg1, 0, len(self.state) - 1)
		new_character = chr(self._lerp(arg2, 0, 255))
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
		new_character = chr(self._lerp(arg2, 0, 255))
		self.state = self.state[:character_pos] + new_character + self.state[character_pos:]

	def _insert_after(self, arg1, arg2):
		character_pos = self._lerp(arg1, 0, len(self.state) - 1)
		new_character = chr(self._lerp(arg2, 0, 255))
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
		return sum([ord(c) for c in self.state])

	def reset(self):
		self.state = self.initial_state
		return self.state
	
	def step(self, action_embedding):
		#print(self.state)
		with torch.no_grad():
			chosen_action = torch.multinomial(torch.softmax(action_embedding[:, :-2] / self.temperature, dim=1), num_samples = 1).item()
			if random.random() < self.epsilon:
				arg1 = torch.sigmoid(action_embedding[:, -2]).item()
			else:
				arg1 = random.uniform(0, 1)
			if random.random() < self.epsilon:
				arg2 = torch.sigmoid(action_embedding[:, -1]).item()
			else:
				arg2 = random.uniform(0, 1)
		self._apply_mutation(chosen_action, arg1, arg2)
		return self.state, self._evaluate_state()
	
def run_env(env, policy, device, training_pipeline, train=False, action_horizion=25):
	next_state = env.reset()
	total_reward = 0

	for _ in range(action_horizion):
		state_embedding = torch.tensor([float(ord(c)) for c in next_state]).unsqueeze(0).unsqueeze(-1).to(device)
		with torch.no_grad():
			action_embedding = policy(state_embedding)
		next_state, reward = env.step(action_embedding)
		next_state_embedding = torch.tensor([float(ord(c)) for c in next_state]).unsqueeze(0).unsqueeze(-1).to(device)
		total_reward = reward
		reward = torch.tensor(reward).to(device)
		if train:
			training_pipeline.record_experience(state_embedding, next_state_embedding, action_embedding, reward)

	if train:
		training_pipeline.replay_experiences()

	return env.state, total_reward


device = 'cuda' if torch.cuda.is_available() else 'cpu'

actor = Actor(
	input_dim=1,
	output_dim=7,
	embedding_dim=16,
	hidden_dim=32,
	encoder_layers=4,
	decoder_layers=4).to(device)

critic = Critic(
	state_input_dim=1,
	action_input_dim=7,
	output_dim=1,
	embedding_dim=16,
	hidden_dim=32,
	encoder_layers=4,
	decoder_layers=4).to(device)

training_pipeline = TrainingPipeline(actor, critic)

environment = ToyEnvironment('ju sun', 0.5, 0.5)
reward_lst = []

for i in range(500):
	final_state, total_reward = run_env(environment, actor, device, training_pipeline, True, 125)
	print(f'Episode: {i}, Final State: {final_state}, Total Reward: {total_reward}')
	reward_lst.append(total_reward)


test_environment = ToyEnvironment('ju sun', 1, 1)

final_state, total_reward = run_env(test_environment, actor, device, training_pipeline, False, 25)
print(f'Final State: {final_state}, Total Reward: {total_reward}')

plt.plot(reward_lst, marker='o')  # Add markers for better visualization
plt.title("Line Chart Example")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)  # Optional: Add a grid
plt.show()