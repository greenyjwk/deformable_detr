import torch

# Load checkpoint
checkpoint_path = "/mnt/storage/ji/Deformable-DETR/TEMP/eval.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# Check what keys are inside
print(checkpoint.keys())
print(checkpoint["counts"])
print()
# print(checkpoint["scores"])