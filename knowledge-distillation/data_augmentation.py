import numpy as np
from typing import List, Tuple
import re
import math

def data_aug(batched_audio: List[np.array], batched_label: List[str], batched_prev: List[str], grain_sec: float = 0.5) -> Tuple[List[np.array], List[str], List[str]]:
    """
    Perform data augmentation on batched audio data by creating shorter segments with corresponding labels.
    For each segment, only include words that are completely within the boundary.
    If a word's end timestamp exceeds the boundary, remove that word and use the boundary timestamp.
    
    Args:
        batched_audio: List of audio arrays (sample_rate=16000)
        batched_label: List of labels with time codes
        batched_prev: List of previous context for each audio
        grain_sec: Time interval for segmentation (default: 0.5 seconds)
    
    Returns:
        Tuple containing augmented audio arrays, labels, and previous contexts
    """
    augmented_batched_audio = []
    augmented_batched_label = []
    augmented_batched_prev = []
    
    SAMPLE_RATE = 16000
    
    # Process each audio sample in the batch
    for audio, label, prev in zip(batched_audio, batched_label, batched_prev):
        # Find the last timestamp in the label
        upper_bound = float(re.findall(r'<\|(\d+\.\d+)\|>', label)[-1])
        
        # Calculate number of complete segments before the upper bound
        # Use floor division to exclude segments that would reach or exceed the upper bound
        time_cnt = int(upper_bound / grain_sec)
        if time_cnt * grain_sec >= upper_bound:
            time_cnt -= 1
        
        # Split the label into tokens (timestamps and text)
        tokens = re.findall(r'(<\|\d+\.\d+\|>|[^<]+|<\|endoftext\|>)', label)
        
        # Process each time segment
        for t in range(time_cnt):
            current_bound = grain_sec * (t + 1)
            current_tokens = []
            
            # Process tokens sequentially
            i = 0
            while i < len(tokens):
                token = tokens[i]
                
                # If it's a timestamp
                if re.match(r'<\|\d+\.\d+\|>', token):
                    timestamp = float(re.findall(r'(\d+\.\d+)', token)[0])
                    
                    # If timestamp exceeds current boundary, break
                    if timestamp > current_bound:
                        break
                    
                    current_tokens.append(token)
                else:
                    current_tokens.append(token)
                
                i += 1
            
            # Add boundary timestamp and endoftext token
            if current_tokens:
                # Remove the last text token if it was incomplete
                if not re.match(r'<\|\d+\.\d+\|>', current_tokens[-1]):
                    current_tokens.pop()
                
                # Add boundary timestamp and endoftext
                current_tokens.append(f"<|{current_bound:.2f}|>")
                current_tokens.append("<|endoftext|>")
                
                # Create augmented label
                augmented_label = "".join(current_tokens)
                
                # Calculate corresponding audio segment
                end_sample = int(current_bound * SAMPLE_RATE)
                audio_segment = audio[:end_sample]
                
                # Add to augmented batches
                augmented_batched_audio.append(audio_segment)
                augmented_batched_label.append(augmented_label)
                augmented_batched_prev.append(prev)
    
    # Add original samples to augmented batch
    augmented_batched_audio.extend(batched_audio)
    augmented_batched_label.extend(batched_label)
    augmented_batched_prev.extend(batched_prev)
    
    return augmented_batched_audio, augmented_batched_label, augmented_batched_prev

if __name__ == "__main__":
    # Sample batch of size 2
    # Assuming sample rate is 16,000 Hz

    # Audio sample 1 (duration: 2.7 seconds)
    audio1 = np.arange(0, int(2.0 * 16000))  # Simple ramp signal
    label1 = "<|0.00|>Hello<|0.23|><|0.23|>world<|1.10|><|1.10|>test<|1.20|><|1.20|><|1.25|><|1.25|>UUUUUUUUU<|2.00|><|endoftext|>"
    prev1 = None

    # Audio sample 2 (duration: 2.0 seconds)
    audio2 = np.arange(0, int(1.92 * 16000))  # Simple ramp signal
    label2 = "<|0.00|>Sample<|0.63|><|0.63|>audio<|1.10|><|1.10|>data<|1.80|><|1.80|>augmentation<|1.92|><|endoftext|>"
    prev2 = None

    # Batch data
    batched_audio = [audio1, audio2]
    batched_label = [label1, label2]
    batched_prev = [prev1, prev2]

    # Apply data augmentation
    aug_audio, aug_label, aug_prev = data_aug(batched_audio, batched_label, batched_prev, grain_sec=0.5)

    print("Original Labels:")
    print("=" * 70)
    print(label1)
    print(label2)
    print("=" * 70)

    # Print the results
    print("\nAugmented Segments:")
    for i in range(len(aug_audio)):
        print(f"\nAugmented Audio {i+1}:")
        print(f"Audio Length (samples): {len(aug_audio[i])}")
        print(f"Label: {aug_label[i]}")
        print(f"Prev: {aug_prev[i]}")
        print("-" * 70)