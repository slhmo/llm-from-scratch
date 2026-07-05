import numpy as np
import torch
from src.Model.attention import MultiHeadAttention
from src.Configs import DTYPE, TEMPERATURE, CONTEXT_WINDOW, DEVICE, \
    W_UP_DIMENSION, N_FEATURES, N_ATTENTION_HEADS



class TransformerBlock:
    def __init__(self, n_features, n_head, context_window, w_up_dimension):
        """
        A single Transformer Layer containing independent Multi-Head Attention
        and a Feed-Forward Network (MLP), complete with residual connections.
        Used in Transformer class
        """
        # unique attention heads for this specific layer
        self.attention = MultiHeadAttention(n_head, n_features, context_window=context_window)

        # unique MLP weights for this specific layer
        self.Wup = torch.randn((n_features, w_up_dimension), dtype=DTYPE, device=DEVICE) * (1.0 / np.sqrt(n_features))      # mlp
        self.Wup.requires_grad_(True)
        self.Wdown = torch.randn((w_up_dimension, n_features), dtype=DTYPE, device=DEVICE) * (1.0 / np.sqrt(w_up_dimension))
        self.Wdown.requires_grad_(True)

        self.params = self.attention.params + [self.Wup, self.Wdown]

    def forward(self, x):
        # Attention Layer + Residual Connection
        x = x + self.attention.forward(x)

        # MLP Feed-Forward Layer + Residual Connection
        x = x + self.mlp(x)
        return x

    def mlp(self, x):
        change = x @ self.Wup     # (n_vectors * wup)
        change = torch.relu(change)
        change = change @ self.Wdown
        return change


class Transformer:
    def __init__(self, vocab_size, n_layers):
        self.vocab_size = vocab_size

        # raw embedding/unembedding matrices
        self.W_embed = torch.randn((vocab_size, N_FEATURES), dtype=DTYPE, device=DEVICE)*(1.0 / np.sqrt(vocab_size))
        self.W_embed.requires_grad_(True)   # rows are tokens, columns are features
        self.W_unembed = torch.randn((N_FEATURES, vocab_size), dtype=DTYPE, device=DEVICE)*(1.0 / np.sqrt(N_FEATURES))
        self.W_unembed.requires_grad_(True) # rows are features, columns are tokens

        # minute deviation from the paper: we use learned positional embedding matrix
        self.W_pos = torch.randn((CONTEXT_WINDOW, N_FEATURES), dtype=DTYPE,device=DEVICE) * (1.0 / np.sqrt(CONTEXT_WINDOW))
        self.W_pos.requires_grad_(True)     # embedding of the positions (learned by training). will be sliced for smaller windows

        # Instantiate a list of independent sequential blocks
        self.blocks = [TransformerBlock(N_FEATURES, N_ATTENTION_HEADS, CONTEXT_WINDOW, W_UP_DIMENSION) for _ in range(n_layers)]

        # gather all params in the entire model for pytorch gradient
        self.params = [self.W_embed, self.W_unembed, self.W_pos]
        for block in self.blocks:
            self.params.extend(block.params)


    def forward(self, token_ids, seq_len):
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long, device=DEVICE)
        else:
            token_ids = token_ids.to(DEVICE)    # (BATCH_SIZE * Block_size)

        # Embed tokens
        x = self.W_embed[token_ids]     # (BATCH_SIZE * Block_size * n_features)

        # Add positional encoding up to the current sequence length
        x = x + self.W_pos[:seq_len]

        # 2. Pass sequentially through all Transformer Blocks
        for block in self.blocks:
            x = block.forward(x)

        # Unembed to get vocabulary logits
        logits = x @ self.W_unembed
        return logits


    def pretrain_step(self, X_tokens, Y_targets, optimizer):
        Y_targets = Y_targets.to(DEVICE)
        optimizer.zero_grad()

        # X_tokens = (BATCH_SIZE * Block_size), seq_len = block size
        logits = self.forward(X_tokens, seq_len=X_tokens.size(1))
        loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab_size), Y_targets.view(-1))
        loss.backward()
        # Adam handles the step update
        optimizer.step()

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
