import torch
import torch.nn.functional as F

def gumbel_softmax_sample(logits, tau=1.0):
	"""
	Sample from the Gumbel-Softmax distribution (differentiable).
	
	Args:
		logits: Tensor of shape [batch_size, num_actions]
		tau: Temperature parameter (controls the smoothness of the distribution)

	Returns:
		Tensor of shape [batch_size, num_actions] representing the sampled actions.
	"""
	# Generate Gumbel noise
	gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
	
	# Add Gumbel noise to logits
	noisy_logits = logits + gumbel_noise

	# Apply softmax with temperature
	return F.softmax(noisy_logits / tau, dim=-1)

# Example usage
logits = torch.tensor([1.2, 0.3, -0.5], requires_grad=True)  # Example logits
print(f'Prob dist: {F.softmax(logits)}')
tau = 1
epsilon = 1
epsilon_decay = 0.999
# for j in range (300):
# 	scores = torch.zeros(3)
# 	epsilon *= epsilon_decay
# 	tau *= 2 - epsilon  # Temperature (lower = more discrete-like)
# 	for i in range(100):
# 		action_one_hot = F.gumbel_softmax(logits, tau=tau, hard=True)
# 		scores += action_one_hot
# 	print(f'Tau: {tau}, epsilon: {epsilon}, scores: {scores}')
scores = torch.zeros(3)
for i in range(100):
	action_one_hot = F.gumbel_softmax(logits, tau=5, hard=True)
	scores += action_one_hot
print(f'Tau: {tau}, epsilon: {epsilon}, scores: {scores}')