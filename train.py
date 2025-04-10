import numpy as np
import random
import os
from vanet_environment import *

# Constants for training
NUM_EPISODES = 1000
NUM_STEPS = 100
EVALUATE_INTERVAL = 100  # Evaluasi setiap 100 episode
MODEL_FILE = 'q_table.npy'  # Nama file untuk menyimpan Q-table

def save_model():
    """Menyimpan model Q-table ke file"""
    print(f"Saving model to {MODEL_FILE}...")
    np.save(MODEL_FILE, q_table)

def load_model():
    """Memuat model Q-table dari file"""
    if os.path.exists(MODEL_FILE):
        print(f"Loading model from {MODEL_FILE}...")
        return np.load(MODEL_FILE)
    else:
        print("No model found, starting with an empty Q-table.")
        return np.zeros_like(q_table)  # Mengembalikan Q-table yang kosong jika tidak ada model

def train():
    total_rewards = []  # To store rewards for evaluation
    
    # Load model (if exists) before training
    global q_table
    q_table = load_model()
    
    for episode in range(NUM_EPISODES):
        state = (random.choice(POWER_BINS), random.choice(BEACON_BINS), random.choice(CBR_BINS))
        episode_reward = 0  # Total reward in this episode
        
        for step in range(NUM_STEPS):
            action = select_action(state)
            new_state = apply_action(state, action)
            reward = calculate_reward(state[2])
            episode_reward += reward  # Accumulate reward
            
            update_q_table(state, action, reward, new_state)
            state = new_state
        
        total_rewards.append(episode_reward)
        
        # Evaluasi setiap EVALUATE_INTERVAL episode
        if episode % EVALUATE_INTERVAL == 0 and episode != 0:
            average_reward = np.mean(total_rewards[-EVALUATE_INTERVAL:])  # Calculate average reward for the last EVALUATE_INTERVAL episodes
            print(f"Evaluation at Episode {episode}: Average Reward: {average_reward}")
    
    # Final evaluation after training
    print("Training completed. Final evaluation...")
    average_reward = np.mean(total_rewards)
    print(f"Average reward over all episodes: {average_reward}")
    
    # Save the model after training
    save_model()
    
    return total_rewards

if __name__ == "__main__":
    train()
