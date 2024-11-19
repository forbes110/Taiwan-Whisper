import json
import os
import argparse
import pandas as pd

def calculate_channel_durations(channel_meta_csv):
    # Read channel metadata
    channels_df = pd.read_csv(channel_meta_csv)
    
    results = []
    for channel in channels_df['channel_list']:
        channel_dir = f"mnt/data/{channel}"  # Update with actual path pattern
        channel_total_sec = 0
        
        # Get JSON files in channel directory
        try:
            json_files = [f for f in os.listdir(channel_dir) if f.endswith('.json')]
            
            for jsonfile in json_files:
                try:
                    with open(os.path.join(channel_dir, jsonfile), 'r') as jf:
                        data = json.load(jf)
                        if "duration" in data:
                            channel_total_sec += data["duration"]
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error processing {jsonfile}: {e}")
                    
            duration_hours = channel_total_sec / 3600
            print(f"{channel} total hours: {duration_hours:.2f}")
            results.append({'channel': channel, 'duration(hr)': duration_hours})
            
        except FileNotFoundError:
            print(f"Directory not found for channel: {channel}")
    
    # Create and save results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('channel_durations.csv', index=False)
    print(f"Total duration across all channels: {results_df['duration(hr)'].sum():.2f} hours")

def main():
    parser = argparse.ArgumentParser(description="Calculate total duration from JSON files in a specified directory.")
    parser.add_argument("channel_meta_csv", type=str, help="Path to CSV file containing channel list")
    args = parser.parse_args()
    
    calculate_channel_durations(args.channel_meta_csv)

if __name__ == "__main__":
    main()