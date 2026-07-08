import os

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

CONTEXT_WINDOW = 256
DTYPE = torch.float32
N_FEATURES = 512
TEMPERATURE = 0.7
N_LAYERS = 12         # number of attention blocks

BATCH_SIZE = 32      # Process 64 sequences at once!
N_EPOCHS = 4       # How many times we want to consume the whole dataset

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


TRAIN_TOKENIZER_PATHS = (PROJECT_ROOT / "data" / "tiny_stories" / "train.csv", )
TOKENIZER_PATH = PROJECT_ROOT / "output" / "Tokenizer" / "tokenizer.pkl"   # save tokenizer on

RAW_TRAIN_PATH = PROJECT_ROOT / "data" / "tiny_stories" / "train.csv"             # will be used to create tokenized train data
TOKENIZED_TRAIN_PATH = PROJECT_ROOT / "output" / "tokenized_train.bin"        # save the tokenized binary data on(will be used to train model)

MODEL_PATH = PROJECT_ROOT / "output" / "Model" / "transformer_weights.pt"      # where model will be stored after pretraining