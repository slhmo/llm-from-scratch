# Initialize your raw model
import torch

from src.Configs import CONTEXT_WINDOW, data_paths, BATCH_SIZE, N_EPOCHS
from src.Model.transformer import Transformer
from src.Tokenizer.letter_tokenizer import Tokenizer

tokenizer = Tokenizer()
tokenizer.tokenize(data_paths)

model = Transformer(vocab_size=tokenizer.vocab_size)

for epoch in range(N_EPOCHS):
    step = 0
    while tokenizer.has_next(step, BATCH_SIZE):
        step +=1
        X_batch, Y_batch = tokenizer.get_next_batch(BATCH_SIZE, CONTEXT_WINDOW)

        loss_value = model.pretrain_step(X_batch, Y_batch, learning_rate=0.01)
        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss_value:.4f}")


# --- PREDICTION MODE ---
prompt = tokenizer.encode("To be, or not to be")
generated_text = model.predict(prompt, max_new_tokens=50, tokenizer=tokenizer)
print(f"Generated text:\n{generated_text}")

# todo: multi headed attention=> outputs must pass through an output projection matrix W_O to bring them back down to the proper variance   -   optimization
# todo: head dimension = n_features/n_head    -    4D Broadcasting Overhead vs. Optimized 2D GEMM