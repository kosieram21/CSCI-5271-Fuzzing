import os 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import gym
import warnings
from torch.distributions import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence
warnings.simplefilter("ignore")

## Initilaize Data store, Actor Network and Critic network



############################# Data Store ####################################################
class PPOMemory():
    """
    Memory for PPO
    """
    def  __init__(self, batch_size):
        self.states = []
        self.actions= []
        self.action_probs = []
        self.rewards = []
        self.vals = []
        self.dones = []
        
        self.batch_size = batch_size

    def generate_batches(self):
        ## suppose n_states=20 and batch_size = 4
        n_states = len(self.states)
        ##n_states should be always greater than batch_size
        ## batch_start is the starting index of every batch
        ## eg:   array([ 0,  4,  8, 12, 16]))
        batch_start = np.arange(0, n_states, self.batch_size) 
        ## random shuffling if indexes
        # eg: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
        indices = np.arange(n_states, dtype=np.int64)
        ## eg: array([12, 17,  6,  7, 10, 11, 15, 13, 18,  9,  8,  4,  3,  0,  2,  5, 14,19,  1, 16])
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        ## eg: [array([12, 17,  6,  7]),array([10, 11, 15, 13]),array([18,  9,  8,  4]),array([3, 0, 2, 5]),array([14, 19,  1, 16])]

        padded_states = pad_sequence([s.squeeze(0).cpu() for s in self.states], batch_first=True)
        return np.array([state.unsqueeze(0) for state in padded_states]),np.array([a.cpu() for a in self.actions]),\
               np.array(self.action_probs),np.array(self.vals),np.array(self.rewards),\
               np.array(self.dones),batches
    
       
    

    def store_memory(self,state,action,action_prob,val,reward,done):
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)
        self.vals.append(val)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions= []
        self.action_probs = []
        self.rewards = []
        self.vals = []
        self.dones = []


############################ Actor Network ######################################

## initialize actor network and critic network


class ActorNwk(nn.Module):
    def __init__(self, num_chars, char_embedding_size, 
                 out_dim,
                 adam_lr,
                 chekpoint_file,
                 hidden1_dim=256,
                 hidden2_dim=256,
                 encoder_embedding_dim=128,
                 encoder_layers=2,
                 action_std_init=0.6
                 ):
        super(ActorNwk, self).__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.out_dim = out_dim
        self.action_std = action_std_init
        self.action_var = torch.full((out_dim,), action_std_init * action_std_init).to(self.device)
        self.code_book = nn.Linear(num_chars, char_embedding_size)
        self.gru = nn.GRU(char_embedding_size, encoder_embedding_dim, encoder_layers, batch_first=True)
        self.actor_nwk = nn.Sequential(
            nn.Linear(encoder_embedding_dim,hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim,hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim,out_dim),  
            nn.Sigmoid()
        )

        self.checkpoint_file = chekpoint_file
        self.optimizer = torch.optim.Adam(params=self.actor_nwk.parameters(),lr=adam_lr)

        
        self.to(self.device)

    def anneal(self):
        self.action_std *= 0.99
        self.action_var = torch.full((self.out_dim,), self.action_std * self.action_std).to(self.device)

    
    def forward(self,state):
        char_embedding = self.code_book(state)
        state_embedding, _ = self.gru(char_embedding)
        state_embedding = state_embedding[:, -1, :]
        action_mean = self.actor_nwk(state_embedding)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        return dist
        #dist = Categorical(out)
        #return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

############################### Crirtic Network ######################################

class CriticNwk(nn.Module):
    def __init__(self,num_chars,
                char_embedding_size,
                 adam_lr,
                 chekpoint_file,
                 hidden1_dim=256,
                 hidden2_dim=256,
                 encoder_embedding_dim=128,
                 encoder_layers=2
                 ):
        super(CriticNwk, self).__init__()

        self.code_book = nn.Linear(num_chars, char_embedding_size)
        self.gru = nn.GRU(char_embedding_size, encoder_embedding_dim, encoder_layers, batch_first=True)
        self.critic_nwk = nn.Sequential(
            nn.Linear(encoder_embedding_dim,hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim,hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim,1),  
   
        )

        self.checkpoint_file = chekpoint_file
        self.optimizer = torch.optim.Adam(params=self.critic_nwk.parameters(),lr=adam_lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    
    def forward(self,state):
        char_embedding = self.code_book(state)
        state_embedding, _ = self.gru(char_embedding)
        state_embedding = state_embedding[:, -1, :]
        out = self.critic_nwk(state_embedding)
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

## Initilaize an Agent will will be able to train the model

############################# Agent ########################################3

## agent

class Agent():
    def __init__(self, gamma, policy_clip,lamda, adam_lr,
                 n_epochs, batch_size, num_chars, char_embedding_size, action_dim):
        
        self.gamma = gamma 
        self.policy_clip = policy_clip
        self.lamda  = lamda
        self.n_epochs = n_epochs

        self.actor = ActorNwk(num_chars=num_chars, char_embedding_size=char_embedding_size, out_dim=action_dim,adam_lr=adam_lr,chekpoint_file='tmp/actor')
        self.critic = CriticNwk(num_chars=num_chars, char_embedding_size=char_embedding_size,adam_lr=adam_lr,chekpoint_file='tmp/ctitic')
        self.memory = PPOMemory(batch_size)

    def store_data(self,state,action,action_prob,val,reward,done):
        self.memory.store_memory(state,action,action_prob,val,reward,done)
       

    def save_models(self):
        print('... Saving Models ......')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_models(self):
        print('... Loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, state):
        #state = torch.tensor([state], dtype=torch.float).to(self.actor.device)
        #state = torch.tensor([float(ord(c)) for c in state], dtype=torch.float).unsqueeze(0).unsqueeze(-1).to(self.actor.device)

        dist = self.actor(state)
        ## sample the output action from a categorical distribution of predicted actions
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        #action = torch.squeeze(action).item()
        #action = action.cpu()

        ## value from critic model
        value = self.critic(state)
        value = torch.squeeze(value).item()

        return action, probs, value
    
    def calculate_advanatage(self,reward_arr,value_arr,dones_arr):
        time_steps = len(reward_arr)
        advantage = np.zeros(len(reward_arr), dtype=np.float32)

        for t in range(0,time_steps-1):
            discount = 1
            running_advantage = 0
            for k in range(t,time_steps-1):
                if int(dones_arr[k]) == 1:
                    running_advantage += reward_arr[k] - value_arr[k]
                else:
                
                    running_advantage += reward_arr[k] + (self.gamma*value_arr[k+1]) - value_arr[k]

                running_advantage = discount * running_advantage
                # running_advantage += discount*(reward_arr[k] + self.gamma*value_arr[k+1]*(1-int(dones_arr[k])) - value_arr[k])
                discount *= self.gamma * self.lamda
            
            advantage[t] = running_advantage
        advantage = torch.tensor(advantage).to(self.actor.device)
        return advantage
    
    def anneal(self):
        self.actor.anneal()
    
    def learn(self):
        for _ in range(self.n_epochs):

            ## initially all will be empty arrays
            state_arr, action_arr, old_prob_arr, value_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()
            
            advantage_arr = self.calculate_advanatage(reward_arr,value_arr,dones_arr)
            values = torch.tensor(value_arr).to(self.actor.device)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).squeeze(0).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage_arr[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage_arr[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage_arr[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

### Toy Environment

import math

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
        reward = 0
        test_string = 'abcdefghijklmnopqrstuvwxyz'
        max_len = min(len(test_string), len(self.state))
        for index in range(max_len):
            reward -= abs(ord(test_string[index]) - ord(self.state[index]))
            if test_string[index] == self.state[index]:
                reward += 100
        for char in set(test_string):
            if char in self.state:
                reward += 50
        if len(self.state) == len(test_string):
            reward += 1000
        return reward
        # reward = 0
        # seen = set()
        # for character in self.state:
        #    if character not in seen:
        #        reward += 3 #ord(character)
        #        seen.add(character)
        # # if len(self.state) > 0 and self.state[0] == '[':
        # # 	reward += 1000
        # reward -= len(self.state)
        # return reward
        #return sum([ord(c) for c in self.state])

    def reset(self):
        self.state = self.initial_state
        return self.state
    
    # def step(self, action_one_hot, arg1, arg2):
    #     # do we detach?
    #     action = torch.argmax(action_one_hot).item()
    #     arg1_val = arg1.item()
    #     arg2_val = arg2.item()
    #     initial_reward = self._evaluate_state()
    #     self._apply_mutation(action, arg1_val, arg2_val)
    #     final_reward = self._evaluate_state()
    #     return self.state, final_reward - initial_reward
    
    def step(self, action):
        # do we detach?
        with torch.no_grad():
            squeezed_action = action.squeeze(0)
            choosen_action = self._lerp(torch.clamp(squeezed_action[0], 0.0, 1.0).item(), 0, 2)
            arg1_val = torch.clamp(squeezed_action[1], 0.0, 1.0).item()
            arg2_val = torch.clamp(squeezed_action[2], 0.0, 1.0).item()
        initial_reward = self._evaluate_state()
        self._apply_mutation(choosen_action, arg1_val, arg2_val)
        final_reward = self._evaluate_state()
        return self.state, final_reward #final_reward - initial_reward

### Train the model



import gym
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def create_one_hot(char, min_ord, max_ord):
    char_ord = ord(char)
    index = char_ord - min_ord
    # with torch.no_grad():
    #     one_hot = torch.zeros(max_ord - min_ord + 1)
    #     one_hot[index] = 1
    one_hot = [0.0] * (max_ord - min_ord + 1)
    one_hot[index] = 1.0
    return one_hot

if not os.path.exists('tmp'):
    os.makedirs('tmp')

#env = gym.make('CartPole-v0')
env = ToyEnvironment('s')
N = 25#20
batch_size = 1#5
n_epochs = 4
alpha = 0.0003
agent = Agent(num_chars = (122-65 + 1),
              char_embedding_size = 128,
              action_dim=3,#env.action_space.n, 
              batch_size=batch_size,
              n_epochs=n_epochs,
              policy_clip=0.2,
              gamma=0.99,lamda=0.95, 
              adam_lr=alpha)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_games = 3000
figure_file = 'cartpole.png'
#best_score = env.reward_range[0]
score_history = []
learn_iters = 0
avg_score = 0
n_steps = 0
max_steps = 125
for i in range(n_games):
    #current_state,info = env.reset()
    current_state = env.reset()
    terminated,truncated = False,False
    done = False
    n_steps=0
    score = 0
    while not done:
        #print(f'state: {env.state}')
        #current_state = torch.tensor([float(ord(c)) for c in current_state]).unsqueeze(0).unsqueeze(-1).to(device)
        current_state = torch.tensor([create_one_hot(char, 65, 122) for char in current_state]).unsqueeze(0).to(device)
        action, prob, val = agent.choose_action(current_state)
        #next_state, reward, terminated, truncated, info = env.step(action)
        next_state, reward = env.step(action)
        #done = 1 if (terminated or truncated) else 0
        n_steps += 1
        done = 1 if n_steps > max_steps else 0
        score += reward
        agent.store_data(current_state, action, prob, val, reward, done)
        if n_steps % N == 0:
            agent.learn()
            agent.anneal()
            learn_iters += 1
        current_state = next_state
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    #if avg_score > best_score:
    #    best_score = avg_score
    #    agent.save_models()
    print(f'state: {env.state}')
    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
            'time_steps', n_steps, 'learning_steps', learn_iters)
    
    
x = [i+1 for i in range(len(score_history))]
plot_learning_curve(x, score_history, figure_file)