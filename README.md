# VANET Q-Learning Project

This project implements Q-Learning for optimizing VANET parameters such as transmission power and beacon rate. The goal is to find the optimal parameters for each vehicle based on the observed Channel Busy Ratio (CBR). The project is modular, allowing for training, plotting of results, and inference through a server that listens for requests from a client (MATLAB).

## Project Structure
vanet_rl_project/ ├── main.py # Entry point for selecting mode (train, plot, or server) ├── train.py # Script to train the Q-Learning model and save the model ├── plot.py # Script to plot reward vs timestep ├── inference_server.py # Server script to receive data from MATLAB and use the model for inference ├── vanet_environment.py # Contains the core Q-Learning logic and environment dynamics ├── requirements.txt # Project dependencies (e.g., numpy, matplotlib) ├── q_table.npy # The trained Q-table saved after training └── README.md # This file 


## Features

1. **Training**  
   Trains the Q-Learning model to optimize VANET parameters (transmission power and beacon rate).  
   The model is saved in the file `q_table.npy` after training.

2. **Plotting**  
   After training, you can generate a plot of reward vs. timestep to visualize model performance.

3. **Inference Server**  
   A server listens to requests from MATLAB clients, loads the trained Q-table, and uses it to compute and return optimized parameters (transmission power, beacon rate, and MCS).

## How to Use

1. **Install Dependencies**  
   Make sure you have installed the required dependencies by running:
   ```bash
   pip install -r requirements.txt

2. **Training**
   python main.py --mode train

3. **Plotting**
   python main.py --mode plot

4. **Inference Server**
   python main.py --mode server


