import flwr as fl
import argparse

parser = argparse.ArgumentParser(description='Distbelief training example')
parser.add_argument("--num_rounds", type=int, default=1)
args = parser.parse_args()

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=args.num_rounds))
