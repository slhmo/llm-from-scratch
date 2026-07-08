import os

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

CONTEXT_WINDOW = 512
DTYPE = torch.float32
N_FEATURES = 512
TEMPERATURE = 0.7
N_LAYERS = 12         # number of attention blocks

BATCH_SIZE = 256      # Process 64 sequences at once!
N_EPOCHS = 8       # How many times we want to consume the whole dataset

W_UP_DIMENSION = 4*N_FEATURES
N_ATTENTION_HEADS = 8
LEARNING_RATE = 0.0006
LR_DECAY = True         # cosine decay with warmups
WARMUP_RATIO = 0.005     # warmup training steps

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCAB_SIZE = 16384
SPECIAL_TOKENS = ["<|endoftext|>"]

# paths:
# TRAIN_TOKENIZER_PATHS = (PROJECT_ROOT / "data" / "tiny_stories" / "sampled_train.csv", )
# TOKENIZER_PATH = PROJECT_ROOT / "src" / "Tokenizer" / "tokenizer.pkl"   # save tokenizer on
#
# TOKENIZED_TRAIN_PATH = PROJECT_ROOT / "data" / "tiny_stories" / "tokenized_train.bin"        # save the tokenized binary data on(will be used to train model)
# RAW_TRAIN_PATH = PROJECT_ROOT / "data" / "tiny_stories" / "sampled_train.csv"             # will be used to create tokenized train data
#
# MODEL_PATH = PROJECT_ROOT / "src" / "Model" / "transformer_weights.pt"      # where model will be stored after pretraining



IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

if IS_KAGGLE:
    # Update this string to match your exact Kaggle dataset folder name
    KAGGLE_DATA_DIR = Path("/kaggle/input/tinystories-tokenized-v1")

    TOKENIZER_PATH = KAGGLE_DATA_DIR / "tokenizer.pkl"
    TOKENIZED_TRAIN_PATH = KAGGLE_DATA_DIR / "tokenized_train.bin"
    RAW_TRAIN_PATH = None
    MODEL_PATH = Path("/kaggle/working/transformer_weights.pt")
else:
    TRAIN_TOKENIZER_PATHS = (PROJECT_ROOT / "data" / "tiny_stories" / "sampled_train.csv",)
    TOKENIZER_PATH = PROJECT_ROOT / "src" / "Tokenizer" / "tokenizer.pkl"
    TOKENIZED_TRAIN_PATH = PROJECT_ROOT / "data" / "tiny_stories" / "tokenized_train.bin"
    RAW_TRAIN_PATH = PROJECT_ROOT / "data" / "tiny_stories" / "train.csv"
    MODEL_PATH = PROJECT_ROOT / "src" / "Model" / "transformer_weights.pt"