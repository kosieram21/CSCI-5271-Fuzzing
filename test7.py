import string
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Mlp(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, final_sigmoid=False):
        super(Mlp, self).__init__()
        dim_list = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        layers = []
        for i in range(len(dim_list) - 1):
            layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            if i < len(dim_list) - 2:
                layers.append(nn.LayerNorm(dim_list[i + 1]))
                layers.append(nn.ReLU())
            elif final_sigmoid:
                layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class Actor(nn.Module): # policy pi
    def __init__(self, 
              embedding_dim, hidden_dim, encoder_layers, decoder_layers, num_actions, 
              tau, epsilon, epsilon_decay, epsilon_min):
        super(Actor, self).__init__()
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gru = nn.GRU(1, embedding_dim, encoder_layers, batch_first=True)
        self.action_decoder = Mlp(embedding_dim, num_actions, hidden_dim, decoder_layers)
        self.arg1_head = Mlp(embedding_dim + num_actions, 1, hidden_dim, decoder_layers, True)
        self.arg2_head = Mlp(embedding_dim + num_actions, 1, hidden_dim, decoder_layers, True)

    def forward(self, state):
        state_embedding, _ = self.gru(state)
        state_embedding = state_embedding[:, -1, :]
        action_logits = self.action_decoder(state_embedding)
        action_one_hot = F.gumbel_softmax(action_logits, tau=self.tau, hard=True, dim=-1) # should we anneal tau?
        joint_embedding = torch.cat((state_embedding, action_one_hot), dim=-1)
        arg1 = self.arg1_head(joint_embedding)
        arg2 = self.arg2_head(joint_embedding)
        noise = torch.rand_like(arg1) * self.epsilon
        arg1 = torch.clamp(arg1 + noise, 0.0, 1.0)
        arg2 = torch.clamp(arg2 + noise, 0.0, 1.0)
        return action_one_hot, arg1, arg2

    def anneal(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Critic(nn.Module): # Q function
    def __init__(self, embedding_dim, hidden_dim, encoder_layers, decoder_layers, num_actions):
        super(Critic, self).__init__()
        self.gru = nn.GRU(1, embedding_dim, encoder_layers, batch_first=True)
        self.mlp = Mlp(embedding_dim + num_actions + 2, 1, hidden_dim, decoder_layers)

    def forward(self, state, action_one_hot, arg1, arg2):
        state_embedding, _ = self.gru(state)
        state_embedding = state_embedding[:, -1, :]
        state_action_space = torch.cat((state_embedding, action_one_hot, arg1, arg2), dim=-1)
        reward = self.mlp(state_action_space)
        return reward
    
class TrainingPipeline:
    def __init__(self, actor, critic, actor_lr=0.001, critic_lr=0.001, gamma=0.99):
        self.memory = [] # potentially this should be dequeue
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma

    def record_experience(self, state, next_state, action_one_hot, arg1, arg2, reward):
        self.memory.append((state, next_state, action_one_hot, arg1, arg2, reward))

    def replay_experiences(self):
        if not self.memory:
            return

        for state, next_state, action_one_hot, arg1, arg2, reward in self.memory:
            with torch.no_grad(): # smelly
                next_action_one_hot, next_arg1, next_arg2 = self.actor(next_state)
                next_q = self.critic(next_state, next_action_one_hot, next_arg1, next_arg2)
                target_q = reward + self.gamma * next_q

            estimated_q = self.critic(state, action_one_hot, arg1, arg2)
            critic_loss = self.criterion(estimated_q, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            action_pred, arg1_pred, arg2_pred = self.actor(state)
            actor_loss = -self.critic(state, action_pred, arg1_pred, arg2_pred)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        self.memory.clear()
        self.actor.anneal()

class ToyEnvironment:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.state = initial_state

    def _lerp(self, input, start, end):
        return math.floor(((end - start) * input) + start)

    def _replace(self, arg1, arg2):
        character_pos = self._lerp(arg1, 0, len(self.state) - 1)
        new_character = chr(self._lerp(arg2, 65, 122))
        state_buff = list(self.state)
        state_buff[character_pos] = new_character
        self.state = ''.join(state_buff)

    def _insert(self, arg1, arg2):
        character_pos = self._lerp(arg1, 0, len(self.state) - 1)
        new_character = chr(self._lerp(arg2, 65, 122))
        self.state = self.state[:character_pos+1] + new_character + self.state[character_pos+1:]

    def _delete(self, arg1, arg2):
        if len(self.state) <= 1:
            return
        character_pos = self._lerp(arg1, 0, len(self.state) - 1)
        self.state = self.state[:character_pos] + self.state[character_pos+1:]

    def _apply_mutation(self, action, arg1, arg2):
        if action == 0:
            self._replace(arg1, arg2)
        elif action == 1:
            self._insert(arg1, arg2)
        elif action == 2:
            self._delete(arg1, arg2)
        else:
            print('we should not be here')

    def _evaluate_state(self):
        return sum([ord(c) for c in self.state])

    def reset(self):
        self.state = self.initial_state
        return self.state
    
    def step(self, action_one_hot, arg1, arg2):
        # do we detach?
        action = torch.argmax(action_one_hot).item()
        arg1_val = arg1.item()
        arg2_val = arg2.item()
        initial_reward = self._evaluate_state()
        self._apply_mutation(action, arg1_val, arg2_val)
        final_reward = self._evaluate_state()
        return self.state, final_reward - initial_reward
    
def run_env(env, actor, device, training_pipeline, train=False, action_horizion=25):
    next_state = env.reset()
    total_reward = 0

    for _ in range(action_horizion):
        state_embedding = torch.tensor([float(ord(c)) for c in next_state]).unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            action_one_hot, arg1, arg2 = actor(state_embedding)
        next_state, reward = env.step(action_one_hot, arg1, arg2)
        total_reward += reward
        next_state_embedding = torch.tensor([float(ord(c)) for c in next_state]).unsqueeze(0).unsqueeze(-1).to(device)
        reward = torch.tensor(reward).to(device)
        if train:
            training_pipeline.record_experience(state_embedding, next_state_embedding, action_one_hot, arg1, arg2, reward)

    if train:
        training_pipeline.replay_experiences()

    return env.state, total_reward

device = 'cuda' if torch.cuda.is_available() else 'cpu'

actor = Actor(
    embedding_dim=16,
    hidden_dim=32,
    encoder_layers=4,
    decoder_layers=4,
    num_actions=3,
    tau=1,
    epsilon=1,
    epsilon_decay=0.99,
    epsilon_min=0.01).to(device)

critic = Critic(
    embedding_dim=16,
    hidden_dim=32,
    encoder_layers=4,
    decoder_layers=4,
    num_actions=3).to(device)

training_pipeline = TrainingPipeline(actor, critic)

environment = ToyEnvironment('seed')
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
