import torch
import numpy as np
from src.Configs import DTYPE, DEVICE


class SingleHeadAttention:
    def __init__(self, n_features, query_size):
        """
        :param n_features: number of features in embedding space
        :param query_size: query vector size
        """
        self.query_size = query_size
        scale_factor = (1.0 / np.sqrt(n_features))
        self.query_matrix = torch.randn((n_features, query_size), dtype=DTYPE, device=DEVICE)*scale_factor
        self.query_matrix.requires_grad_(True)
        self.key_matrix = torch.randn((n_features, query_size), dtype=DTYPE, device=DEVICE)*scale_factor
        self.key_matrix.requires_grad_(True)
        self.value_up_matrix = torch.randn((n_features, query_size), dtype=DTYPE, device=DEVICE)*scale_factor
        self.value_up_matrix.requires_grad_(True)
        self.value_down_matrix = torch.randn((query_size, n_features), dtype=DTYPE, device=DEVICE)*(1.0 / np.sqrt(query_size))    # efficient. 3b1b recommended
        self.value_down_matrix.requires_grad_(True)

        self.params = [self.query_matrix, self.key_matrix, self.value_up_matrix, self.value_down_matrix]

    def forward(self, input_vectors):
        # input_sequence shape: (seq_len, n_features)
        scores = (input_vectors @ self.query_matrix) @ (input_vectors @ self.key_matrix).transpose(-2, -1) # to support batch. Shape: (batch_size, seq_len, seq_len)
        mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attention_pattern = torch.softmax(scores / np.sqrt(self.query_size), dim=-1)
        value_vectors = input_vectors @ self.value_up_matrix @ self.value_down_matrix
        attention_output = attention_pattern @ value_vectors
        return attention_output


# class MultiHeadAttention:
#     def __init__(self, n_head, n_features, query_size):
#         """
#         Multi-Head Attention implemented as a list of independent SingleHeadAttention heads for readability.
#         Each head independently reads from and adds to the residual stream.
#         """
#         self.n_head = n_head
#         # Instantiate independent SingleHeadAttention blocks
#         self.heads = [SingleHeadAttention(n_features, query_size) for _ in range(n_head)]
#
#         # flatten all head parameters into a single list so transformer's SGD loop tracks them
#         self.params = []
#         for head in self.heads:
#             self.params.extend(head.params)
#
#     def forward(self, input_vectors):
#         # input_vectors shape: (batch_size, seq_len, n_features)
#         # sum up the outputs of all individual attention heads
#         return sum(head.forward(input_vectors) for head in self.heads)


# class MultiHeadAttention:
#     def __init__(self, n_head, n_features, query_size, context_window):
#         """
#         Vectorized Multi-Head Attention that runs all heads in parallel using
#         PyTorch's tensor broadcasting.
#         """
#         self.n_head = n_head
#         self.query_size = query_size
#
#         scale_factor = (1.0 / np.sqrt(n_features))
#
#         # matrices shaped with an extra dimension for the heads: (n_head, n_features, query_size)
#         self.query_matrix = torch.randn((n_head, n_features, query_size), dtype=DTYPE, device=DEVICE) * scale_factor
#         self.query_matrix.requires_grad_(True)
#
#         self.key_matrix = torch.randn((n_head, n_features, query_size), dtype=DTYPE, device=DEVICE) * scale_factor
#         self.key_matrix.requires_grad_(True)
#
#         self.value_up_matrix = torch.randn((n_head, n_features, query_size), dtype=DTYPE, device=DEVICE) * scale_factor
#         self.value_up_matrix.requires_grad_(True)
#
#         self.value_down_matrix = torch.randn((n_head, query_size, n_features), dtype=DTYPE, device=DEVICE) * ( 1.0 / np.sqrt(query_size))
#         self.value_down_matrix.requires_grad_(True)
#
#         mask = torch.full((context_window, context_window), float('-inf'), device=DEVICE)
#         self.mask = torch.triu(mask, diagonal=1) # Lower triangle is 0.0, upper is -inf
#
#         # Track all parameters for SGD optimization step
#         self.params = [self.query_matrix, self.key_matrix, self.value_up_matrix, self.value_down_matrix]
#
#     def forward(self, input_vectors):
#         # input_vectors shape: (batch_size, seq_len, n_features)
#
#         # introduce the head dimension: (batch_size, 1, seq_len, n_features)(currently 1, will broadcast to n_head)
#         x = input_vectors.unsqueeze(1)
#
#         # x @ self.query_matrix: project across all heads simultaneously using matrix multiplication broadcasting
#         # key/query matrices: (batch_size, 1, seq_len, n_features) @ (n_head, n_features, query_size) -> (batch_size, n_head, seq_len, query_size)
#
#         # compute attention scores: (batch_size, n_head, seq_len, query_size) @ (batch_size, n_head, query_size, seq_len)
#         # -> (batch_size, n_head, seq_len, seq_len)
#         scores = (x @ self.query_matrix) @ ((x @ self.key_matrix).transpose(-2, -1))
#
#         # mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()     # inefficient
#         seq_len = scores.size(-1)
#
#         scores = scores + self.mask[:seq_len, :seq_len]
#
#         attention_pattern = torch.softmax(scores / np.sqrt(self.query_size), dim=-1)
#
#         # calculate value vectors: (batch_size, 1, seq_len, n_features) @ (n_head, n_features, query_size) @ (n_head, query_size, n_features)
#         # -> (batch_size, n_head, seq_len, n_features)
#         value_vectors = x @ self.value_up_matrix @ self.value_down_matrix
#
#         # apply attention pattern to value vectors: (batch_size, n_head, seq_len, seq_len) @ (batch_size, n_head, seq_len, n_features)
#         # -> (batch_size, n_head, seq_len, n_features)
#         attention_output = attention_pattern @ value_vectors
#
#         # sum the outputs of all heads to produce the final change matrix for the residual stream
#         # -> (batch_size, seq_len, n_features)
#         return attention_output.mean(dim=1)


class MultiHeadAttention:
    def __init__(self, n_head, n_features, context_window):
        self.n_head = n_head
        self.n_features = n_features
        self.head_dim = n_features // n_head        # query_size

        # ensure n_features divides perfectly by n_head
        assert n_features % n_head == 0, "n_features must be divisible by n_head"

        # combined linear projections for all heads at once
        self.q_proj = torch.randn((n_features, n_features), dtype=DTYPE, device=DEVICE) * (1.0 / np.sqrt(n_features))
        self.k_proj = torch.randn((n_features, n_features), dtype=DTYPE, device=DEVICE) * (1.0 / np.sqrt(n_features))
        self.v_proj = torch.randn((n_features, n_features), dtype=DTYPE, device=DEVICE) * (1.0 / np.sqrt(n_features))

        # W_O matrix to project concatenated heads back to the residual stream
        self.out_proj = torch.randn((n_features, n_features), dtype=DTYPE, device=DEVICE) * (1.0 / np.sqrt(n_features))

        self.q_proj.requires_grad_(True)
        self.k_proj.requires_grad_(True)
        self.v_proj.requires_grad_(True)
        self.out_proj.requires_grad_(True)

        mask = torch.full((context_window, context_window), float('-inf'), device=DEVICE)
        self.mask = torch.triu(mask, diagonal=1)

        self.params = [self.q_proj, self.k_proj, self.v_proj, self.out_proj]

    def forward(self, input_vectors):
        B, T, C = input_vectors.shape  # Batch size, Sequence length, n_features (256)

        # Project inputs and reshape into (B, n_head, T, head_dim)
        q = (input_vectors @ self.q_proj).view(B, T, self.n_head,
                                               self.head_dim).transpose(1, 2)
        k = (input_vectors @ self.k_proj).view(B, T, self.n_head,
                                               self.head_dim).transpose(1, 2)
        v = (input_vectors @ self.v_proj).view(B, T, self.n_head,
                                               self.head_dim).transpose(1, 2)

        # Compute Attention Scores across all heads simultaneously
        scores = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = scores + self.mask[:T, :T]
        attention_pattern = torch.softmax(scores, dim=-1)

        # Apply weights to values: (B, n_head, T, head_dim)
        attention_output = attention_pattern @ v

        # Concatenate heads back together: (B, T, C)
        # transpose makes it contiguous, view collapses the head and head_dim dimensions back into C
        out = attention_output.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection to mix head representations back into the residual stream
        return out @ self.out_proj  # outputs must pass through an output projection matrix W_O to bring them back down to the proper variance