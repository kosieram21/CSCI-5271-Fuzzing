import os
import struct
import torch
import torch.nn as nn
import math
import random

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
	
class FuzzingEnvironment:
	def __init__(self, epsilon, epsilon_decay):
		self.state = None
		self.next_state = None
		self.reward = None
		self.next_reward = None
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.memory = []

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

	def reset(self):
		self.state = self.initial_state
		self.epsilon *= self.epsilon_decay
		return self.state
	
	def step(self, action_embedding):
		with torch.no_grad():
			chosen_action = torch.multinomial(torch.softmax(action_embedding[:, :-2] / (self.epsilon + 0.5), dim=1), num_samples = 1).item()
			if random.random() < (1 - self.epsilon):
				arg1 = torch.sigmoid(action_embedding[:, -2]).item()
			else:
				arg1 = random.uniform(0, 1)
			if random.random() < (1 - self.epsilon):
				arg2 = torch.sigmoid(action_embedding[:, -1]).item()
			else:
				arg2 = random.uniform(0, 1)
		self._apply_mutation(chosen_action, arg1, arg2)
		return self.state, self.next_reward - self.reward
	
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
			actor_loss = -self.critic(state, predicted_action)
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

		self.memory.clear()

FIFO_C_TO_PY = 'c_to_py_fifo'
FIFO_PY_TO_C = 'py_to_c_fifo'

class ModelInterface():
	def __init__(self):
		self.readPipe = None
		self.writePipe = None

	def open(self):
		if not os.path.exists(FIFO_C_TO_PY):
			os.mkfifo(FIFO_C_TO_PY)
		if not os.path.exists(FIFO_PY_TO_C):
			os.mkfifo(FIFO_PY_TO_C)

		self.readPipe = open(FIFO_C_TO_PY, 'rb')
		self.writePipe = open(FIFO_PY_TO_C, 'wb')

	def close(self):
		self.readPipe.close()
		self.writePipe.close()

	def receive_command(self):
		header = self.readPipe.read(4)
		payload_size = struct.unpack('<I', header)[0]
		payload = self.readPipe.read(payload_size)
		command, args = payload.decode('utf-8').split(':', 1)
		return command, args.encode('utf-8')

	def send_response(self, payload):
		payload_size = len(payload)
		header = payload_size.to_bytes(4, byteorder='little')
		self.writePipe.write(header)
		self.writePipe.write(payload)
		self.writePipe.flush()

model_interface = ModelInterface()
model_interface.open()
processing = False

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

while processing:
	command, args = model_interface.receive_command()

	if command == 'Close':
		print('closing...')
		model_interface.send_response((0).to_bytes(1, byteorder='little'))
		model_interface.close()
		processing = False
	elif command == 'GetAction':
		print('getting action...')
		state_size = struct.unpack('<I', args[:4])[0]
		state = args[4:].decode('utf-8')
		print(f'state size: {state_size}')
		print(f'state: {state}')
		try:
			error_code = 0
			action = 3  #number corresponding to model output of what action to take
		except:
			error_code = 1
			action = 0  #do nothing, we failed
		response_payload = (error_code).to_bytes(1, byteorder='little')
		response_payload += (action).to_bytes(1, byteorder='little')
		model_interface.send_response(response_payload)
	elif command == 'RecordExperience':
		print('recording experience...')
		state_size = struct.unpack('<I', args[:4])[0]
		state = args[4: 4 + state_size].decode('utf-8')
		next_state_size = struct.unpack('<I', args[4 + state_size: 8 + state_size])[0]
		next_state = args[8 + state_size: 8 + state_size + next_state_size].decode('utf-8')
		action = int.from_bytes(args[8 + state_size + next_state_size: 9 + state_size + next_state_size], byteorder='little', signed=False)
		reward = struct.unpack('<I', args[9 + state_size + next_state_size: 13 + state_size + next_state_size])[0]
		print(f'state size: {state_size}')
		print(f'state: {state}')
		print(f'next state size: {next_state_size}')
		print(f'next state: {next_state}')
		print(f'action: {action}')
		print(f'reward: {reward}')
		model_interface.send_response((0).to_bytes(1, byteorder='little'))
	elif command == 'ReplayExperiences':
		print('replaying experiences...')
		training_pipeline.replay_experiences()
		model_interface.send_response((0).to_bytes(1, byteorder='little'))
