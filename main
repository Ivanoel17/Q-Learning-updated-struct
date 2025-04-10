import argparse
from train import train
from plot import plot_rewards
from inference_server import QLearningServer

def main():
    parser = argparse.ArgumentParser(description="Choose an option")
    parser.add_argument('--mode', choices=['train', 'plot', 'server'], required=True, help="Choose the mode: 'train', 'plot', or 'server'")

    args = parser.parse_args()

    if args.mode == 'train':
        print("Training started...")  # Keterangan saat menjalankan training
        train()
        print("Training completed.")
        
    elif args.mode == 'plot':
        print("Plotting started...")  # Keterangan saat menjalankan plotting
        plot_rewards()
        print("Plotting completed.")
        
    elif args.mode == 'server':
        print("Inference server started...")  # Keterangan saat menjalankan server
        server = QLearningServer()
        server.start()
        print("Inference server stopped.")
        
if __name__ == "__main__":
    main()
