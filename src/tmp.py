import torch

from src.Configs import CONTEXT_WINDOW, BATCH_SIZE, N_EPOCHS, \
    LEARNING_RATE, VOCAB_SIZE, TOKENIZER_PATH, data_paths, BLOCK_SIZE
from src.Model.optimizers import CustomAdam
from src.Model.transformer import Transformer
from src.Tokenizer.BPE_tokenizer import BPETokenizer
from src.Tokenizer.letter_tokenizer import Tokenizer

tokenizer = BPETokenizer(VOCAB_SIZE)
tokenizer.load(TOKENIZER_PATH)
# tokenizer = Tokenizer()
# tokenizer.tokenize(paths_to_data=data_paths)



model = Transformer(vocab_size=tokenizer.vocab_size)
optimizer = CustomAdam(model.params, lr=LEARNING_RATE)


for epoch in range(N_EPOCHS):
    step = 0
    for path in data_paths:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

            print(f"Tokenizing {path}...")
            all_tokens = tokenizer.encode(text)
            chunk_size = BLOCK_SIZE * BATCH_SIZE

            for i in range(0, len(all_tokens) - chunk_size-1, chunk_size):
                s = torch.tensor(all_tokens[i:i+chunk_size+1], dtype=torch.long)        # shape: chunk size list of ids
                # shape: (BATCH_SIZE * Block_size) each row is a block of data to be processed
                x = torch.stack([s[j * BLOCK_SIZE: (j + 1) * BLOCK_SIZE] for j in range(BATCH_SIZE)])
                y = torch.stack([s[j * BLOCK_SIZE + 1: (j + 1) * BLOCK_SIZE + 1] for j in range(BATCH_SIZE)])   # shape: (BATCH_SIZE * Block_size)

                loss_value = model.pretrain_step(x, y, optimizer=optimizer)
                if step % 100 == 0:
                    print(f"Epoch: {epoch} | Step {step} | Loss: {loss_value:.4f}")


# import torch.autograd.profiler as profiler
# X_batch, Y_batch = tokenizer.get_next_batch(BATCH_SIZE, CONTEXT_WINDOW)
# with profiler.profile(use_device='cuda') as prof:
#     model.pretrain_step(X_batch, Y_batch, optimizer=optimizer)
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# --- PREDICTION MODE ---
prompt = tokenizer.encode("To be, or not to be")
generated_text = model.predict(prompt, max_new_tokens=50, tokenizer=tokenizer)
print(f"Generated text:\n{generated_text}")

# todo: multi headed attention=> outputs must pass through an output projection matrix W_O to bring them back down to the proper variance
# todo: optimization(torch implementation)? - head dimension = n_features/n_head    -    4D Broadcasting Overhead vs. Optimized 2D GEMM