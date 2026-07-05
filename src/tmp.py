import random

import torch

from src.Configs import CONTEXT_WINDOW, BATCH_SIZE, N_EPOCHS, \
    LEARNING_RATE, VOCAB_SIZE, TOKENIZER_PATH, data_paths, BLOCK_SIZE, N_LAYERS
from src.Model.optimizers import CustomAdam
from src.Model.transformer import Transformer
from src.Tokenizer.BPE_tokenizer import BPETokenizer
from src.Tokenizer.letter_tokenizer import Tokenizer


def main():
    tokenizer = BPETokenizer(VOCAB_SIZE)
    tokenizer.load(TOKENIZER_PATH)
    # tokenizer = Tokenizer()
    # tokenizer.tokenize(paths_to_data=data_paths)



    model = Transformer(vocab_size=tokenizer.vocab_size, n_layers=N_LAYERS)
    optimizer = CustomAdam(model.params, lr=LEARNING_RATE)

    print("Pre-tokenizing datasets into memory...")
    pre_tokenized_data = {}
    for path in data_paths:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Tokenizing {path}...")
        pre_tokenized_data[path] = tokenizer.encode(text)

    for epoch in range(N_EPOCHS):
        step = 0
        for path in data_paths:
            all_tokens_tensor = torch.tensor(pre_tokenized_data[path], dtype=torch.long)
            chunk_size = BLOCK_SIZE * BATCH_SIZE

            # we chop the entire dataset into distinct, non-overlapping blocks
            num_blocks = (len(all_tokens_tensor) - 1) // BLOCK_SIZE
            # this creates a 2D matrix => shape: [num_blocks, BLOCK_SIZE]
            X_all = all_tokens_tensor[:num_blocks * BLOCK_SIZE].view(num_blocks, BLOCK_SIZE)
            Y_all = all_tokens_tensor[1:num_blocks * BLOCK_SIZE + 1].view(num_blocks, BLOCK_SIZE)
            # shuffle the row indices
            block_indices = torch.randperm(num_blocks)

            # shuffle blocks indices
            block_indices = list(range(num_blocks))
            random.shuffle(block_indices)

            # now we have diverse batches
            for i in range(0, num_blocks, BATCH_SIZE):
                batch_idx = block_indices[i: i + BATCH_SIZE]
                if len(batch_idx) < BATCH_SIZE:
                    break
                x = X_all[batch_idx]
                y = Y_all[batch_idx]
                loss_value = model.pretrain_step(x, y, optimizer=optimizer)
                step += 1
                if step % 10 == 0:
                    print(f"Epoch: {epoch} | Step {step} | Loss: {loss_value:.4f}")


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



# todo: multi headed attention=> outputs must pass through an output projection matrix W_O to bring them back down to the proper variance??
# todo: optimization(torch implementation)?    -    4D Broadcasting Overhead vs. Optimized 2D GEMM
# todo: add tokenizer splitting, end-of-text and etc