"""dummy development file. ignore this"""

# embedding matrix: n(vocab)*n(features)
import numpy as np
from src.Tokenizer.letter_tokenizer import Tokenizer
import torch
import torch.nn.functional as F


CONTEXT_WINDOW = 128
DTYPE = torch.float32
N_FEATURES = 64
QUERY_SIZE = 32
i = 0   # i'th chunk from input

# in this example 64 vocab_size, arbitrary 64 features
tokenizer = Tokenizer()
tokenizer.tokenize(('../../data/Tiny-Shakespeare.txt', ))

embedding_matrix = torch.randn((tokenizer.vocab_size, N_FEATURES), dtype=DTYPE, requires_grad=True)
inputs = tokenizer.divide(batch_size=CONTEXT_WINDOW+1)
current_text_chunk = inputs[i]
token_ids = tokenizer.encode(current_text_chunk)

X_tokens = token_ids[:-1]  # The prompt (Length: CONTEXT_WINDOW)
# PyTorch requires Long tensors (integers) for calculating cross-entropy loss targets
Y_targets = torch.tensor(token_ids[1:], dtype=torch.long)

input_vectors = embedding_matrix[X_tokens]  # Shape: (CONTEXT_WINDOW, N_FEATURES)

# dot product each key and query and do normalization(softmax) for each embedding's values

# query matrix @ each E(i)  => query vector for that E. (E => Embedded letter)
# query matrix: (n_features*arbitrary query size).
query_matrix = torch.randn((N_FEATURES, QUERY_SIZE), dtype=DTYPE, requires_grad=True)
# for each vocab, query vector is E(i)@query_matrix   => query_vectors matrix = E@query_matrix where query_vectors[i] is query vector for E(i)


# key matrix: (n_features*arbitrary query size).
key_matrix = torch.randn((N_FEATURES, QUERY_SIZE), dtype=DTYPE, requires_grad=True)
# for each vocab, key vector is E(i)@key_matrix   => key_vectors matrix = E@key_matrix where key_vectors[i] is key vector for E(i)



# then if we do a = query_vectors@key_vectors.T  a[i, :] will be queries of E(i) answered. we need to apply softmax to entries of each row
# tip: divide by root of dimension of key-query matrices as recommended by the paper.
scores = (input_vectors @ query_matrix) @ (input_vectors @ key_matrix).T
# we want to apply parallelized training where we don't train only on the last token but all the preceding tokens also, so we need to prevent allowing later tokens
# to effect earlier ones, so we have to set the upper triangular piece of scores to -inf so after softmax it becomes 0 and query[i] get 0 key match for key[i+j]
mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
attention_pattern = torch.softmax(scores / np.sqrt(QUERY_SIZE), dim=-1)


# we now define Value matrix. we do value_vectors=input_sequence@value_matrix and therefor value_vectors[i, :]
# represents the effect E[i] would have ON OTHER EMBEDDINGS, if it were relevant
# value_matrix = torch.randn((N_FEATURES, N_FEATURES), dtype=DTYPE, requires_grad=True) # bad, naive approach
value_up_matrix = torch.randn((N_FEATURES, QUERY_SIZE), dtype=DTYPE, requires_grad=True)
value_down_matrix = torch.randn((QUERY_SIZE, N_FEATURES), dtype=DTYPE, requires_grad=True)  # more efficient. 3b1b recommended

value_vectors = input_vectors @ value_up_matrix @ value_down_matrix
# now for each E[i] we do E[i] += sum(value_vectors[j, :] * relevance of E[i] and E[j] )which essentially
# tells for any other embedding find the effect it has on embedding[i], multiply by the relevance and add to the embedding[i] which is done by multiplication
attention_output = attention_pattern @ value_vectors
input_vectors = input_vectors + attention_output


# we would repeat previous steps + multilayer perceptron but let's assume we already did
# multiply E[i] by unembedding_matrix, search in the embedding_matrix for the closest vector to each row, decode and done
unembedding_matrix = torch.randn((N_FEATURES, tokenizer.vocab_size), dtype=DTYPE, requires_grad=True)
unembedding_results = input_vectors @ unembedding_matrix


loss = F.cross_entropy(unembedding_results, Y_targets)
# This single line calculates the gradients for ALL matrices
loss.backward()


# final search and decode
result = list()
predicted_token_ids = torch.argmax(unembedding_results, dim=-1).tolist()
predicted_text = tokenizer.decode(predicted_token_ids)

print(f"Original Input Text (Truncated):\n{ ''.join(current_text_chunk[:50]) }...\n")
print(f"Model Predictions (Untrained/Random):\n{ predicted_text[:50] }...")


# --- Add this at the very bottom ---
learning_rate = 0.01

with torch.no_grad():
    # Update the matrices using the calculated gradients (.grad)
    embedding_matrix -= learning_rate * embedding_matrix.grad
    query_matrix     -= learning_rate * query_matrix.grad
    key_matrix       -= learning_rate * key_matrix.grad
    value_up_matrix     -= learning_rate * value_up_matrix.grad
    value_down_matrix     -= learning_rate * value_down_matrix.grad
    unembedding_matrix -= learning_rate * unembedding_matrix.grad

    # Crucial: Reset gradients to zero so they don't pile up in the next loop iteration
    embedding_matrix.grad.zero_()
    query_matrix.grad.zero_()
    key_matrix.grad.zero_()
    value_up_matrix.grad.zero_()
    value_down_matrix.grad.zero_()
    unembedding_matrix.grad.zero_()

print(f"Loss after this iteration: {loss.item():.4f}")