import numpy as np
import torch
from src.Model.attention import MultiHeadAttention
from src.Configs import DTYPE, TEMPERATURE, CONTEXT_WINDOW, DEVICE, \
    W_UP_DIMENSION, N_FEATURES, QUERY_SIZE, N_ATTENTION_HEADS
import torch.nn.functional as F


class Transformer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

        # raw embedding/unembedding matrices
        self.W_embed = torch.randn((vocab_size, N_FEATURES), dtype=DTYPE, device=DEVICE)*(1.0 / np.sqrt(vocab_size))
        self.W_embed.requires_grad_(True)
        self.W_unembed = torch.randn((N_FEATURES, vocab_size), dtype=DTYPE, device=DEVICE)*(1.0 / np.sqrt(N_FEATURES))
        self.W_unembed.requires_grad_(True)
        # minute deviation from the paper: we use learned positional embedding matrix
        self.W_pos = torch.randn((CONTEXT_WINDOW, N_FEATURES), dtype=DTYPE,device=DEVICE) * (1.0 / np.sqrt(CONTEXT_WINDOW))
        self.W_pos.requires_grad_(True)

        self.attention = MultiHeadAttention(N_ATTENTION_HEADS, N_FEATURES, QUERY_SIZE)

        self.Wup = torch.randn((N_FEATURES, W_UP_DIMENSION), dtype=DTYPE, device=DEVICE)*(1.0 / np.sqrt(N_FEATURES))    # mlp
        self.Wup.requires_grad_(True)
        self.Wdown = torch.randn((W_UP_DIMENSION, N_FEATURES), dtype=DTYPE, device=DEVICE)*(1.0 / np.sqrt(W_UP_DIMENSION))
        self.Wdown.requires_grad_(True)

        # gather all params in the entire model for pytorch gradient
        self.params = [self.W_embed, self.W_unembed, self.W_pos, self.Wup, self.Wdown] + self.attention.params


    def forward(self, token_ids, seq_len):
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long, device=DEVICE)
        else:
            token_ids = token_ids.to(DEVICE)

        # 1. Embed tokens
        x = self.W_embed[token_ids]

        # Add positional encoding up to the current sequence length
        x = x + self.W_pos[:seq_len]

        # 2. Attention + Residual connection
        x = x + self.attention.forward(x)

        # NOTE: Your future MLP block will go right here (e.g., x = x + self.mlp.forward(x))
        x = x + self.mlp(x)

        # 3. Unembed to get vocabulary logits
        logits = x @ self.W_unembed
        return logits


    def mlp(self, x):
        change = x@self.Wup      # (n_vectors * wup)
        change = torch.relu(change)
        change = change @ self.Wdown
        return change


    def pretrain_step(self, X_tokens, Y_targets, learning_rate):
        """Processes a chunk of text, computes loss, and updates weights."""
        Y_targets = Y_targets.to(DEVICE)
        # 1. Forward pass
        logits = self.forward(X_tokens, seq_len=CONTEXT_WINDOW)     # train on context_window sized chunks

        # 2. Compute loss Flatten dimensions to (Batch_size * Seq_len, Vocab_size) for cross_entropy
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), Y_targets.view(-1))

        # 3. Backward pass (calculates .grad for all params)
        loss.backward()

        # 4. Manual SGD weight update step
        with torch.no_grad():
            for p in self.params:
                p -= learning_rate * p.grad
                p.grad.zero_()  # Reset for next time

        return loss.item()

    def predict(self, start_tokens, max_new_tokens, tokenizer):
        """Generates new text auto-regressively, token by token."""
        # We don't want PyTorch tracking history/gradients during prediction
        with torch.no_grad():
            context = list(start_tokens)

            for _ in range(max_new_tokens):
                # Ensure we don't exceed our max context window size
                input_ctx = context[-CONTEXT_WINDOW:]
                # Wrap input_ctx in a list to add a batch dimension -> shape: (1, seq_len)
                token_tensor = torch.tensor([input_ctx], dtype=torch.long, device=DEVICE)

                # Get predictions for the current context
                # FIX 2: Set seq_len to the actual size of the token_tensor, not the growing context list
                logits = self.forward(token_tensor, seq_len=token_tensor.size(-1))

                # We only care about the prediction for the VERY LAST token(first dimension is batch which is essentially just 1 in prediction)
                last_token_logits = logits[0, -1, :]
                probs = torch.softmax(last_token_logits/TEMPERATURE, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                # Append the prediction back into the context to predict the next one
                context.append(next_token_id)

        return tokenizer.decode(context)