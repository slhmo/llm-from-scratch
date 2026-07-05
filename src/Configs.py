import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

CONTEXT_WINDOW = 256
BLOCK_SIZE = CONTEXT_WINDOW    # block size for training steps
DTYPE = torch.float32
N_FEATURES = 256
TEMPERATURE = 0.7
N_LAYERS = 4        # number of attention blocks

BATCH_SIZE = 64  # Process 64 sequences at once!
N_EPOCHS = 100     # How many times we want to consume the whole dataset

W_UP_DIMENSION = 2*N_FEATURES
N_ATTENTION_HEADS = 8
LEARNING_RATE = 0.0005

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_paths = (PROJECT_ROOT / "data" / "Tiny-Shakespeare.txt", )
TOKENIZER_PATH = PROJECT_ROOT / "src" / "Tokenizer" / "tokenizer.pkl"

VOCAB_SIZE = 1024
