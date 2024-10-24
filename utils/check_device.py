import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Specify the first GPU (ID=0)
else:
    device = torch.device("cpu")  # Use CPU if no GPU is available

# Print the selected device
print(f"Using device: {device}")

# Optional: Print GPU name if CUDA is available
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Optional: Print the current CUDA device index
print(f"Current CUDA device index: {torch.cuda.current_device()}")

# Allocate a tensor on the selected device
tensor = torch.rand(2, 3).to(device)

# Print the device of the tensor
print(f"Tensor is on device: {tensor.device}")

x = torch.rand(5, 3).to('cuda')
print(x)
