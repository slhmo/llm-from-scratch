import torch
print("CUDA Available:", torch.cuda.is_available())
print("Current Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))