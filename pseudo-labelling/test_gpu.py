# import torch

# def check_cuda_devices():
#     print(f"PyTorch Version: {torch.__version__}")
#     print(f"CUDA Version: {torch.version.cuda}")
#     print(f"CUDA Available: {torch.cuda.is_available()}")
#     if torch.cuda.is_available():
#         num_gpus = torch.cuda.device_count()
#         print(f"Number of GPUs detected by PyTorch: {num_gpus}")
#         for i in range(num_gpus):
#             print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# if __name__ == "__main__":
#     check_cuda_devices()

import torch

# 檢查 PyTorch 是否可用
print("PyTorch 可用性:", torch.__version__)

# 檢查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用 GPU:", torch.cuda.get_device_name)
else:
    device = torch.device("cpu")
    print("使用 CPU")

# 創建一個張量並移動到相
