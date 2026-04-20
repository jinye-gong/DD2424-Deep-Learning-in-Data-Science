import torch

# 查看 GPU 型号
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    
    # 查看 PyTorch 编译时使用的 CUDA 版本
    print(f"PyTorch 使用的 CUDA 版本: {torch.version.cuda}")
    
    # 查看当前显卡的计算能力 (Compute Capability)
    prop = torch.cuda.get_device_properties(0)
    print(f"显存总量: {prop.total_memory / 1024**2:.2f} MB")
else:
    print("未检测到可用 GPU")