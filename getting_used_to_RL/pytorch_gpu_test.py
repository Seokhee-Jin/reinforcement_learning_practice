import torch

"""
Torch version should be 11.6:
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
"""
print(torch.cuda.is_available())