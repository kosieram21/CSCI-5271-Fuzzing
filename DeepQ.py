import os
import struct
import torch
import torch.nn as nn

class DeepQ(nn.Module):
	def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim, encoder_layers, decoder_layers):
		super(DeepQ, self).__init__()
		self.gru = nn.GRU(input_dim, embedding_dim, encoder_layers, batch_first=True)
		dim_list = [embedding_dim] + [hidden_dim] * decoder_layers + [output_dim]
		layers = []
		for i in range(len(dim_list) - 1):
			layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
			if i < len(dim_list) - 2:
				layers.append(nn.ReLU())
		self.mlp = nn.Sequential(*layers)

	def forward(self, state):
		embedding, _ = self.gru(state)
		embedding = embedding[:, -1, :]
		q_scores = self.mlp(embedding)
		return q_scores

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

#model_interface = ModelInterface()
#model_interface.open()
processing = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DeepQ(
	input_dim=1,
	output_dim=5,
	embedding_dim=16,
	hidden_dim=32,
	encoder_layers=4,
	decoder_layers=4).to(device)

input = 'did we line up the legos?'
state = torch.tensor([float(ord(c)) for c in input]).unsqueeze(0).unsqueeze(-1).to(device)
print(state.shape)
q_scores = model(state)
print(q_scores)

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
		model = 0   #update model loop goes here
		model_interface.send_response((0).to_bytes(1, byteorder='little'))
