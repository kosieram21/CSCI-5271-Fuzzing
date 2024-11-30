import random
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import math
import warnings
from torch.distributions import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence
warnings.simplefilter("ignore")
from ppo_fuzzing_model import Agent
from ppo_fuzzing_model import create_one_hot
import subprocess
import time

def get_per_seed_coverage(input_buf):
    """Get coverage signal for the current input buffer using afl-showmap."""
    with open("current_input", "wb") as temp_input:
        temp_input.write(input_buf)

    result = subprocess.run(
        ["afl-showmap", "-o", "coverage_map.txt", "--", "./test_program", "current_input"],
        capture_output=True,
        text=False
    )

    # Parse the coverage map for the number of edges
    with open("coverage_map.txt", "r") as coverage_file:
        edges = len(coverage_file.readlines())
    
    with open("code_coverage.txt", 'a') as f:
        f.write(f'{str(edges)}\n')

    return edges

rl_model = None

def _lerp(input, start, end):
    return math.floor(((end - start) * input) + start)

def _replace(buf, arg1, arg2):
    character_pos = _lerp(arg1, 0, len(buf) - 1)
    new_character = _lerp(arg2, 0, 255)
    buf[character_pos] = new_character
    return bytes(buf)

def _insert(buf, max_len, arg1, arg2):
    character_pos = _lerp(arg1, 0, len(buf) - 1)
    new_character = _lerp(arg2, 0, 255)
    buf.insert(character_pos + 1, new_character)
    return bytes(buf)

def _delete(buf, arg1, arg2):
    if len(buf) <= 1:
        return
    character_pos = _lerp(arg1, 0, len(buf)- 1)
    buf.pop(character_pos)
    return bytes(buf)
    

def _apply_mutation(state, max_len, action, arg1, arg2):
    if action == 0:
        return _replace(state, arg1, arg2)
    elif action == 1:
        return _insert(state, max_len, arg1, arg2)
    elif action == 2:
        return _delete(state, arg1, arg2)
    else:
        print('we should not be here')

def init(seed):
    with open("python_file.txt", 'w') as file:
        pass
    with open("python_file.txt", 'a') as file:
        file.write('You made it to init\n')
    with open('base_seed.txt', 'r') as file:
        base_seed = file.readline()
    with open("input/seed.txt", 'w') as f: #remove all other inputs from the queue and replace with the mutated buffer.
        f.write(base_seed)
    
    
    global rl_model, device

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    random.seed(seed)

    batch_size = 1#5
    n_epochs = 4
    alpha = 0.0003
    rl_model = Agent(num_chars = 256,
              char_embedding_size = 128,
              action_dim=3,#env.action_space.n, 
              batch_size=batch_size,
              n_epochs=n_epochs,
              policy_clip=0.2,
              gamma=0.99,lamda=0.95, 
              adam_lr=alpha)
    
    if os.path.exists("/tmp/actor"):
        rl_model.load_models()

    return 0


def fuzz(buf, add_buf, max_len):
    with open("python_file.txt", 'a') as file:
        file.write('You made it to fuzz\n')
    with open("python_file.txt", 'ab') as file:
        file.write(buf)
    
    global rl_model, device

    current_state = torch.tensor([create_one_hot(char, 0, 255) for char in buf]).unsqueeze(0).to(device)
    action, prob, val = rl_model.choose_action(current_state)
    with torch.no_grad():
        squeezed_action = action.squeeze(0)
        choosen_action = _lerp(torch.clamp(squeezed_action[0], 0.0, 1.0).item(), 0, 2)
        arg1_val = torch.clamp(squeezed_action[1], 0.0, 1.0).item()
        arg2_val = torch.clamp(squeezed_action[2], 0.0, 1.0).item()
    
    buf = _apply_mutation(buf, max_len, choosen_action, arg1_val, arg2_val)
    
    code_coverage = get_per_seed_coverage(buf)
    #code_coverage = 0

    rl_model.store_data(current_state, action, prob, val, code_coverage, False)
    with open("python_file.txt", 'ab') as file:
        file.write(buf)
    with open("input/seed.txt", 'wb') as f: #remove all other inputs from the queue and replace with the mutated buffer.
        f.write(buf)

    return buf[:max_len]


# def describe(max_description_length):
#     # with open("python_file.txt", 'a') as file:
#     #     file.write('You made it to describe\n')
#     return "You made it to describe"[:max_description_length]


def deinit():
    with open("python_file.txt", 'a') as file:
        file.write('You made it to deinit\n')
    
    global rl_model

    rl_model.learn()
    rl_model.anneal()

    rl_model.save_models()

    pass