import os
from collections import defaultdict
import csv

# Initialize a dictionary to store channel -> files mapping
channel_files = defaultdict(list)

# Process all files
base_path = "/mnt/home/ntuspeechlabtaipei1/forbes/validator_inference"
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.startswith("validator_inference"):
            # Get the channel ID from the path
            channel_id = os.path.basename(os.path.dirname(os.path.join(root, file)))
            channel_files[channel_id].append(file)

# Create CSV with channels that have exactly one validator_inference.txt file
with open('validated_channel_names.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(['channel_name'])
    
    # Write channel IDs that meet our criteria
    for channel, files in channel_files.items():
        if len(files) == 1 and files[0] == "validator_inference.txt":
            writer.writerow([channel])