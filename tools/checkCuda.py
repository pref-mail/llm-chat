import torch
print(f"PyTorch版本: {torch. version }")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU名称: {torch.cuda.get_device_name(0)}")