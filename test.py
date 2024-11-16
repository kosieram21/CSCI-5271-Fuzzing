import random
import string
import torch
import torch.nn as nn

class DeepQ(nn.Module):
	def __init__(self, 
			  learning_rate, gamma, epsilon, epsilon_min, epsilon_decay, batch_size, 
			  input_dim, output_dim, embedding_dim, hidden_dim, encoder_layers, decoder_layers):
		super(DeepQ, self).__init__()
		self.memory = []
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay
		self.batch_size = batch_size
		self.gru = nn.GRU(input_dim, embedding_dim, encoder_layers, batch_first=True)
		dim_list = [embedding_dim] + [hidden_dim] * decoder_layers + [output_dim]
		layers = []
		for i in range(len(dim_list) - 1):
			layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
			if i < len(dim_list) - 2:
				layers.append(nn.ReLU())
		self.mlp = nn.Sequential(*layers)

	def record_experience(self, state, next_state, action, reward):
		self.memory.append((state, next_state, action, reward))

	def replay_experiences(self):
		random.shuffle(self.memory)
		num_batches = len(self.memory) // self.batch_size
		batches = [self.memory[i*self.batch_size:(i+1)*self.batch_size] for i in range(num_batches)]
		batches.append(self.memory[num_batches*self.batch_size:])
		for batch in batches:
			state = torch.stack([item[0] for item in batch])
			next_state = torch.stack([item[1] for item in batch])
			action = torch.stack([item[2] for item in batch])
			reward = torch.stack([item[3] for item in batch])
			target = reward + self.gamma * torch.argmax(self.forward(next_state), dim=-1)
			targetQ = self.forward(state)

		self.memory.clear()

	def forward(self, state):
		embedding, _ = self.gru(state)
		embedding = embedding[:, -1, :]
		q_scores = self.mlp(embedding)
		return q_scores
	
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DeepQ(
	learning_rate=0.001,
	gamma=0.95,
	epsilon=1.0,
	epsilon_min=0.1,
	epsilon_decay=0.995,
	batch_size=2,#25,
	input_dim=1,
	output_dim=5,
	embedding_dim=16,
	hidden_dim=32,
	encoder_layers=4,
	decoder_layers=4).to(device)

input = 'did we line up the legos?'
state = torch.tensor([float(ord(c)) for c in input]).unsqueeze(0).unsqueeze(-1).to(device)
#q_scores = model(state)
#print(q_scores)

for i in range(7):
	state = ''.join(random.choices(string.ascii_letters, k=10))
	state = torch.tensor([float(ord(c)) for c in state]).to(device)
	next_state = ''.join(random.choices(string.ascii_letters, k=10))
	next_state = torch.tensor([float(ord(c)) for c in next_state]).to(device)
	action = torch.tensor(random.randint(0, 4)).to(device)
	reward = torch.tensor(random.uniform(0, 100)).to(device)
	model.record_experience(state, next_state, action, reward)

model.replay_experiences()