import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

CONTEXT_WINDOW = 128
DTYPE = torch.float32
N_FEATURES = 64
QUERY_SIZE = 32
TEMPERATURE = 0.7

N_MULTI_HEADED_ATTENTION = 10
BATCH_SIZE = 128  # Process 128 sequences at once!
N_EPOCHS = 1     # How many times we want to consume the whole dataset

W_UP_DIMENSION = 4*N_FEATURES
N_ATTENTION_HEADS = 2

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_paths = (PROJECT_ROOT / "data" / "Tiny-Shakespeare.txt", )
