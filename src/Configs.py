import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

CONTEXT_WINDOW = 256
BLOCK_SIZE = CONTEXT_WINDOW    # block size for training steps
DTYPE = torch.float32
N_FEATURES = 512
TEMPERATURE = 0.7
N_LAYERS = 8         # number of attention blocks

BATCH_SIZE = 64      # Process 64 sequences at once!
N_EPOCHS = 5       # How many times we want to consume the whole dataset

W_UP_DIMENSION = 4*N_FEATURES
N_ATTENTION_HEADS = 8
LEARNING_RATE = 0.0005

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCAB_SIZE = 4096
SPECIAL_TOKENS = ["<|endoftext|>"]

# paths:
TRAIN_TOKENIZER_PATHS = (PROJECT_ROOT / "data" / "tiny_stories" / "sampled_train.csv", )
TOKENIZER_PATH = PROJECT_ROOT / "src" / "Tokenizer" / "tokenizer.pkl"   # save tokenizer on

TOKENIZED_TRAIN_PATH = PROJECT_ROOT / "data" / "tiny_stories" / "tokenized_train.bin"        # save the tokenized binary data on(will be used to train model)
RAW_TRAIN_PATH = PROJECT_ROOT / "data" / "tiny_stories" / "sampled_train.csv"             # will be used to create tokenized train data
