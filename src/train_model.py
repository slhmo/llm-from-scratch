import os

import numpy as np
import torch

from src.Configs import CONTEXT_WINDOW, BATCH_SIZE, N_EPOCHS, \
    LEARNING_RATE, VOCAB_SIZE, TOKENIZER_PATH, TRAIN_TOKENIZER_PATHS, \
    BLOCK_SIZE, N_LAYERS, TOKENIZED_TRAIN_PATH, DEVICE
from src.Model.optimizers import CustomAdam
from src.Model.transformer import Transformer
from src.Tokenizer.BPE_tokenizer import BPETokenizer


def main():
    # tensorFloat32 core acceleration for faster matrix multiplications
    torch.set_float32_matmul_precision('high')
    tokenizer = BPETokenizer(VOCAB_SIZE)
    tokenizer.load(TOKENIZER_PATH)

    model = Transformer(vocab_size=VOCAB_SIZE, n_layers=N_LAYERS)
    # Target the bound method directly and use max-autotune
    model.forward = torch.compile(model.forward, mode="default")

    # optimizer = CustomAdam(model.params, lr=LEARNING_RATE)
    optimizer = torch.optim.AdamW(model.params, lr=LEARNING_RATE, weight_decay=0.01)

    data_dtype = np.uint16
    file_size_bytes = os.path.getsize(TOKENIZED_TRAIN_PATH)
    num_tokens = file_size_bytes // np.dtype(data_dtype).itemsize

    print(f"Loading memory-mapped dataset. Total tokens: {num_tokens:,}")
    binary_dataset = np.memmap(TOKENIZED_TRAIN_PATH, dtype=data_dtype, mode='r')

    for epoch in range(N_EPOCHS):

        prompt = tokenizer.encode("To be, or not to be")
        generated_text = model.predict(prompt, max_new_tokens=50, tokenizer=tokenizer)
        print(f"\n --------Generated text--------:\n{generated_text}")

        max_idx = num_tokens - BLOCK_SIZE - 1
        steps_per_epoch = num_tokens // (BATCH_SIZE * BLOCK_SIZE)

        chunk_size = BLOCK_SIZE * BATCH_SIZE
        # now we have diverse batches
        for step in range(steps_per_epoch):
            ix = torch.randint(0, max_idx, (BATCH_SIZE,))
            # Pull tokens out of the memory map into PyTorch tensors
            x_list = [torch.from_numpy((binary_dataset[i:i + BLOCK_SIZE]).astype(np.int64)) for i in ix]
            y_list = [torch.from_numpy((binary_dataset[i + 1:i + 1 + BLOCK_SIZE]).astype(np.int64)) for i in ix]

            x = torch.stack(x_list).to(DEVICE)
            y = torch.stack(y_list).to(DEVICE)

            loss_value = model.pretrain_step(x, y, optimizer=optimizer)

            if step % 10 == 0:
                print(f"Epoch: {epoch} | Step {step}/{steps_per_epoch} | Loss: {loss_value:.4f}")


    # --- PREDICTION MODE ---
    prompt = tokenizer.encode("To be, or not to be")
    generated_text = model.predict(prompt, max_new_tokens=50, tokenizer=tokenizer)
    print(f"Generated text:\n{generated_text}")

    print("\n" + "=" * 50)
    print("Training complete! Entering interactive generation mode.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    while True:
        user_prompt = input("\nEnter your prompt: ")

        # Check if the user wants to exit
        if user_prompt.strip().lower() in ['exit', 'quit']:
            print("Exiting. Goodbye!")
            break

        # Skip empty inputs
        if not user_prompt.strip():
            print("Prompt cannot be empty. Please try again.")
            continue

        # Encode, predict, and display
        prompt_tokens = tokenizer.encode(user_prompt)

        # You can adjust max_new_tokens or even make it dynamic if you want!
        generated_text = model.predict(prompt_tokens, max_new_tokens=100, tokenizer=tokenizer)

        print("\n--- Model Output ---")
        print(generated_text)
        print("-" * 20)


if __name__ == '__main__':
    main()



# todo: optimization(torch implementation)?    -    4D Broadcasting Overhead vs. Optimized 2D GEMM