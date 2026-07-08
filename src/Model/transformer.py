import numpy as np
import torch
from src.Model.attention import MultiHeadAttention
from src.Configs import DTYPE, TEMPERATURE, CONTEXT_WINDOW, DEVICE, \
    W_UP_DIMENSION, N_FEATURES, N_ATTENTION_HEADS


class CustomLayerNorm:
    def __init__(self, n_features, eps=1e-5):       # => introduces 2*n_features parameters
        self.eps = eps

        # gamma initialized to 1s will be learned
        self.gamma = torch.ones(n_features, dtype=DTYPE, device=DEVICE)
        self.gamma.requires_grad_(True)

        # Learnable shift (beta) initialized to 0s
        self.beta = torch.zeros(n_features, dtype=DTYPE, device=DEVICE)
        self.beta.requires_grad_(True)

        # Expose parameters for optimizer tracking
        self.params = [self.gamma, self.beta]

    def forward(self, x):
        # Compute mean and variance along the last dimension (n_features)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        # Standardize and scale/shift
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class TransformerBlock:
    def __init__(self, n_features, n_head, context_window, w_up_dimension):
        """
        A single Transformer Layer containing independent Multi-Head Attention
        and a Feed-Forward Network (MLP), complete with residual connections.
        Used in Transformer class
        """

        # Layer norms for pre-attention and pre-mlp blocks
        self.ln1 = CustomLayerNorm(n_features)
        self.ln2 = CustomLayerNorm(n_features)

        # unique attention heads for this specific layer
        self.attention = MultiHeadAttention(n_head, n_features, context_window=context_window)

        # unique MLP weights for this specific layer
        self.Wup = torch.randn((n_features, w_up_dimension), dtype=DTYPE, device=DEVICE) * (1.0 / np.sqrt(n_features))      # mlp
        self.Wup.requires_grad_(True)
        self.Wdown = torch.randn((w_up_dimension, n_features), dtype=DTYPE, device=DEVICE) * (1.0 / np.sqrt(w_up_dimension))
        self.Wdown.requires_grad_(True)

        self.params = self.attention.params + [self.Wup, self.Wdown] + self.ln1.params + self.ln2.params


    def forward(self, x):
        # Attention Layer + Residual Connection
        x = x + self.attention.forward(self.ln1.forward(x))

        # MLP Feed-Forward Layer + Residual Connection
        x = x + self.mlp(self.ln2.forward(x))
        return x

    def mlp(self, x):
        change = x @ self.Wup     # (n_vectors * wup)
        change = torch.nn.functional.gelu(change)
        change = change @ self.Wdown
        return change


class Transformer:
    def __init__(self, vocab_size, n_layers):
        self.vocab_size = vocab_size

        # raw embedding/unembedding matrices
        self.W_embed = torch.randn((vocab_size, N_FEATURES), dtype=DTYPE, device=DEVICE)*(1.0 / np.sqrt(vocab_size))
        self.W_embed.requires_grad_(True)   # rows are tokens, columns are features

        # W_unembed => W_embed.T
        # self.W_unembed = torch.randn((N_FEATURES, vocab_size), dtype=DTYPE, device=DEVICE)*(1.0 / np.sqrt(N_FEATURES))
        # self.W_unembed.requires_grad_(True) # rows are features, columns are tokens
        # final LayerNorm before unembedding projection
        self.ln_f = CustomLayerNorm(N_FEATURES)

        # minute deviation from the paper: we use learned positional embedding matrix
        self.W_pos = torch.randn((CONTEXT_WINDOW, N_FEATURES), dtype=DTYPE,device=DEVICE) * (1.0 / np.sqrt(CONTEXT_WINDOW))
        self.W_pos.requires_grad_(True)     # embedding of the positions (learned by training). will be sliced for smaller windows

        # Instantiate a list of independent sequential blocks
        self.blocks = [TransformerBlock(N_FEATURES, N_ATTENTION_HEADS, CONTEXT_WINDOW, W_UP_DIMENSION) for _ in range(n_layers)]

        # gather all params in the entire model for pytorch gradient
        self.params = [self.W_embed, self.W_pos] + self.ln_f.params
        for block in self.blocks:
            self.params.extend(block.params)

        # torch.compile will struggle with the dynamic, changing sequence lengths
        # by having 2 forwards one for train and 1 for prediction we can be sure that train's forward method always gets tensors of static size and we can optimize
        self.eager_forward = self.forward


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

        # Apply final normalization layer
        x = self.ln_f.forward(x)

        # Unembed to get vocabulary logits
        logits = x @ self.W_embed.T
        return logits


    def pretrain_step(self, X_tokens, Y_targets, optimizer, scheduler):
        Y_targets = Y_targets.to(DEVICE)
        X_tokens = X_tokens.to(DEVICE)
        optimizer.zero_grad()

        device_type = 'cuda' if 'cuda' in str(DEVICE) else 'cpu'
        # X_tokens = (BATCH_SIZE * Block_size), seq_len = block size
        # Run the heavy matrix multiplications in fast 16-bit precision
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):     # remove if your gpu doesn't support bfloat16
            logits = self.forward(X_tokens, seq_len=X_tokens.size(1))
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab_size), Y_targets.view(-1))

        # logits = self.forward(X_tokens, seq_len=X_tokens.size(1))
        # loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab_size), Y_targets.view(-1))
        loss.backward()
        # Adam handles the step update
        optimizer.step()
        scheduler.step()

        return loss.item()

    def predict(self, start_tokens, max_new_tokens, tokenizer):
        """Generates new text auto-regressively, token by token."""
        # We don't want PyTorch tracking history/gradients during prediction
        with torch.no_grad():
            context = list(start_tokens)
            eot_id = tokenizer.vocab_special_tokens["<|endoftext|>"]

            for _ in range(max_new_tokens):
                # Ensure we don't exceed our max context window size
                input_ctx = context[-CONTEXT_WINDOW:]
                # Wrap input_ctx in a list to add a batch dimension -> shape: (1, seq_len)
                token_tensor = torch.tensor([input_ctx], dtype=torch.long, device=DEVICE)

                # Get predictions for the current context
                # FIX 2: Set seq_len to the actual size of the token_tensor, not the growing context list
                logits = self.eager_forward(token_tensor, seq_len=token_tensor.size(-1))

                # We only care about the prediction for the VERY LAST token(first dimension is batch which is essentially just 1 in prediction)
                last_token_logits = logits[0, -1, :]
                probs = torch.softmax(last_token_logits/TEMPERATURE, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                if next_token_id == eot_id:
                    break
                # Append the prediction back into the context to predict the next one
                context.append(next_token_id)

        return tokenizer.decode(context)

    def save_weights(self, file_path):
        """save weights to disk"""
        state = {
            'W_embed': self.W_embed.data.cpu(),
            'W_pos': self.W_pos.data.cpu(),
            'ln_f': {
                'gamma': self.ln_f.gamma.data.cpu(),
                'beta': self.ln_f.beta.data.cpu()
            },
            'blocks': [
                {
                    'ln1': {'gamma': b.ln1.gamma.data.cpu(),
                            'beta': b.ln1.beta.data.cpu()},
                    'ln2': {'gamma': b.ln2.gamma.data.cpu(),
                            'beta': b.ln2.beta.data.cpu()},
                    'q_proj': b.attention.q_proj.data.cpu(),
                    'k_proj': b.attention.k_proj.data.cpu(),
                    'v_proj': b.attention.v_proj.data.cpu(),
                    'out_proj': b.attention.out_proj.data.cpu(),
                    'Wup': b.Wup.data.cpu(),
                    'Wdown': b.Wdown.data.cpu(),
                } for b in self.blocks
            ]
        }
        torch.save(state, file_path)
        print(f"saved weights to {file_path}")

    def load_weights(self, file_path):
        """Loads and updates weights"""
        import os
        if not os.path.exists(file_path):
            print(f"wrong path: {file_path}")
            return

        state = torch.load(file_path, map_location=DEVICE)
        with torch.no_grad():
            self.W_embed.copy_(state['W_embed'])
            self.W_pos.copy_(state['W_pos'])
            self.ln_f.gamma.copy_(state['ln_f']['gamma'])
            self.ln_f.beta.copy_(state['ln_f']['beta'])

            for i, b in enumerate(self.blocks):
                b_state = state['blocks'][i]
                b.ln1.gamma.copy_(b_state['ln1']['gamma'])
                b.ln1.beta.copy_(b_state['ln1']['beta'])
                b.ln2.gamma.copy_(b_state['ln2']['gamma'])
                b.ln2.beta.copy_(b_state['ln2']['beta'])
                b.attention.q_proj.copy_(b_state['q_proj'])
                b.attention.k_proj.copy_(b_state['k_proj'])
                b.attention.v_proj.copy_(b_state['v_proj'])
                b.attention.out_proj.copy_(b_state['out_proj'])
                b.Wup.copy_(b_state['Wup'])
                b.Wdown.copy_(b_state['Wdown'])

        print(f"Successfully loaded weights from {file_path}!")
