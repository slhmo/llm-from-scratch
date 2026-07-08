import os

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

from src.Configs import CONTEXT_WINDOW, BATCH_SIZE, N_EPOCHS, \
    LEARNING_RATE, VOCAB_SIZE, TOKENIZER_PATH, \
    N_LAYERS, TOKENIZED_TRAIN_PATH, DEVICE, MODEL_PATH, LR_DECAY, WARMUP_RATIO
from src.Model.optimizers import CustomAdam
from src.Model.transformer import Transformer
from src.Tokenizer.BPE_tokenizer import BPETokenizer


def main():
    data_dtype = np.uint16
    file_size_bytes = os.path.getsize(TOKENIZED_TRAIN_PATH)
    num_tokens = file_size_bytes // np.dtype(data_dtype).itemsize
    steps_per_epoch = num_tokens // (BATCH_SIZE * CONTEXT_WINDOW)

    # tensorFloat32 core acceleration for faster matrix multiplications
    torch.set_float32_matmul_precision('high')
    tokenizer = BPETokenizer(VOCAB_SIZE)
    tokenizer.load(TOKENIZER_PATH)

    model = Transformer(vocab_size=VOCAB_SIZE, n_layers=N_LAYERS)
    model.forward = torch.compile(model.forward, mode="default")

    # optimizer = CustomAdam(model.params, lr=LEARNING_RATE)
    optimizer = torch.optim.AdamW(model.params, lr=LEARNING_RATE, weight_decay=0.01)

    # cosine decay with warmups
    total_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    # the cosine decay for the remaining steps
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps), eta_min=1e-6)   # min LR = 1e-6
    # combine
    if LR_DECAY:
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    else:
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=total_steps)


    print(f"Loading memory-mapped dataset. Total tokens: {num_tokens:,}")
    binary_dataset = np.memmap(TOKENIZED_TRAIN_PATH, dtype=data_dtype, mode='r')

    for epoch in range(N_EPOCHS):
        prompt = tokenizer.encode("To be, or not to be")
        generated_text = model.predict(prompt, max_new_tokens=25, tokenizer=tokenizer)
        print(f"\nepoch: {epoch}\n --------Generated text--------:\n{generated_text}")

        max_idx = num_tokens - CONTEXT_WINDOW - 1

        chunk_size = CONTEXT_WINDOW * BATCH_SIZE
        # now we have diverse batches
        for step in range(steps_per_epoch):
            ix = torch.randint(0, max_idx, (BATCH_SIZE,))
            # Pull tokens out of the memory map into PyTorch tensors
            x_list = [torch.from_numpy((binary_dataset[i:i + CONTEXT_WINDOW]).astype(np.int64)) for i in ix]
            y_list = [torch.from_numpy((binary_dataset[i + 1:i + 1 + CONTEXT_WINDOW]).astype(np.int64)) for i in ix]

            x = torch.stack(x_list).to(DEVICE)
            y = torch.stack(y_list).to(DEVICE)

            loss_value = model.pretrain_step(x, y, optimizer=optimizer, scheduler=scheduler)

            if step % 10 == 0:
                print(f"Epoch: {epoch} | Step {step}/{steps_per_epoch} | Loss: {loss_value:.4f}")

    model.save_weights(MODEL_PATH)

if __name__ == '__main__':
    main()



# todo: KV-Cache