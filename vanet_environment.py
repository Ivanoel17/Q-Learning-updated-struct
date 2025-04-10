import numpy as np
import random

# Constants
CBR_TARGET = 0.65
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 0.1  # Fixed exploration rate

# Simplified state discretization
POWER_BINS = [5, 15, 25, 30]
BEACON_BINS = [1, 5, 10, 20]
CBR_BINS = [0.0, 0.3, 0.6, 1.0]

# Initialize Q-table
q_table = np.zeros((len(POWER_BINS), len(BEACON_BINS), len(CBR_BINS), 2))

def discretize(value, bins):
    return np.digitize(value, bins) - 1

def calculate_reward(cbr):
    return -abs(cbr - CBR_TARGET) * 100  # Simple reward based on CBR deviation

def select_action(state):
    power_idx = discretize(state[0], POWER_BINS)
    beacon_idx = discretize(state[1], BEACON_BINS)
    cbr_idx = discretize(state[2], CBR_BINS)
    
    if random.random() < EPSILON:
        return random.choice([0, 1])  # 0: decrease, 1: increase
    return np.argmax(q_table[power_idx, beacon_idx, cbr_idx])

def update_q_table(state, action, reward, new_state):
    old_idx = discretize(state[0], POWER_BINS), discretize(state[1], BEACON_BINS), discretize(state[2], CBR_BINS)
    new_idx = discretize(new_state[0], POWER_BINS), discretize(new_state[1], BEACON_BINS), discretize(new_state[2], CBR_BINS)
    
    old_q = q_table[old_idx + (action,)]
    max_new_q = np.max(q_table[new_idx])
    q_table[old_idx + (action,)] = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_new_q - old_q)

def apply_action(state, action):
    new_power = max(5, min(30, state[0] + (-1 if action == 0 else 1)))
    new_beacon = max(1, min(20, state[1] + (-1 if action == 0 else 1)))
    return new_power, new_beacon
