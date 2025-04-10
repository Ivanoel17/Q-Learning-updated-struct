import socket
import json
import numpy as np
import os
from vanet_environment import *

HOST = '127.0.0.1'
PORT = 5000
MODEL_FILE = 'q_table.npy'  # Nama file model yang disimpan

class QLearningServer:
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((HOST, PORT))
        self.server.listen(1)
        print(f"Server listening on {HOST}:{PORT}")
        
        # Memuat model yang sudah dilatih
        self.q_table = self.load_model()

    def load_model(self):
        """Memuat model Q-table dari file"""
        if os.path.exists(MODEL_FILE):
            print(f"Loading model from {MODEL_FILE}...")
            return np.load(MODEL_FILE)
        else:
            print("No model found, starting with an empty Q-table.")
            return np.zeros_like(q_table)  # Mengembalikan Q-table yang kosong jika tidak ada model

    def handle_client(self, conn):
        while True:
            data = conn.recv(1024)
            if not data:
                break

            try:
                rl_data = json.loads(data.decode())
                veh_id = rl_data['vehID']
                current_power = rl_data['transmissionPower']
                current_beacon = rl_data['beaconRate']
                current_cbr = rl_data['CBR']
                current_mcs = rl_data['MCS']
                
                print(f"Received data from {veh_id}: Power: {current_power}, Beacon: {current_beacon}, CBR: {current_cbr}, MCS: {current_mcs}")

                # Select action based on the current state
                action = select_action((current_power, current_beacon, current_cbr))
                new_power, new_beacon = apply_action((current_power, current_beacon, current_cbr), action)

                # Calculate the reward
                reward = calculate_reward(current_cbr)

                # Send back optimized parameters
                response = {
                    'power': new_power,
                    'beacon': new_beacon,
                    'MCS': current_mcs,  # MCS is returned as is
                    'reward': reward
                }

                conn.send(json.dumps(response).encode())
                print(f"Sent response: {response}")

            except Exception as e:
                print(f"Error: {e}")
                break

    def start(self):
        while True:
            conn, addr = self.server.accept()
            print(f"Connected to: {addr}")
            self.handle_client(conn)
            conn.close()

if __name__ == "__main__":
    server = QLearningServer()
    server.start()
