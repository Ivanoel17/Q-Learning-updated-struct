import matplotlib.pyplot as plt
import numpy as np
from train import train  # To access total_rewards after training

def plot_rewards():
    print("Generating reward vs timestep plot...")
    
    # Retrieving rewards after training
    rewards = train()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(rewards)), rewards, label="Reward per Episode")
    plt.title("Reward vs Timestep")
    plt.xlabel("Timestep (Episode)")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    plot_rewards()
