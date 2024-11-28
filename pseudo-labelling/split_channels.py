import argparse
import os
import pandas as pd
from math import ceil

def split_channels(directory, num_nodes, csv_path=None, output_dir="./"):
    if csv_path:
        channels_df = pd.read_csv(csv_path)
        channels = channels_df['channel_name'].tolist()
    else:
        channels = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    channels_per_node = ceil(len(channels) / num_nodes)
    
    for i in range(num_nodes):
        start_idx = i * channels_per_node
        end_idx = min((i + 1) * channels_per_node, len(channels))
        node_channels = pd.DataFrame(channels[start_idx:end_idx], columns=['channel_name'])
        output_file = os.path.join(output_dir, f'node_{i}.csv')
        node_channels.to_csv(output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description='Split channels into node CSVs')
    parser.add_argument('--directory', default=None, type=str, help='Directory containing channel subdirectories')
    parser.add_argument('--num_nodes', type=int, help='Number of nodes to split into')
    parser.add_argument('--all_channels_csv', type=str, help='CSV file with channel_name column')
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    
    split_channels(args.directory, args.num_nodes, args.all_channels_csv, args.output_dir)

if __name__ == '__main__':
    main()
    
# python3 split_channels.py --directory /mnt/home/ntuspeechlabtaipei1/tw_separated --num_nodes 3
# python3 split_channels.py --num_nodes 3 --all_channels_csv ./all_channels_tw.csv --output_dir ./