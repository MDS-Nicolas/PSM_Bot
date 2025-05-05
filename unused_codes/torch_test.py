import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Built with CUDA version: {torch.version.cuda}")
print(f"Number of GPUs detected: {torch.cuda.device_count()}")
# Optional: Try a simple CUDA operation
try:
    a = torch.tensor([1.0, 2.0]).cuda()
    print(f"Simple CUDA tensor creation successful: {a}")
except Exception as e:
    print(f"Simple CUDA tensor creation failed: {e}")
exit()