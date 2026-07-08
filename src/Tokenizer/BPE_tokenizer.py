import collections
import pickle
from typing import List, Dict, Tuple        # increase readability
import regex as re

class BPETokenizer:
    def __init__(self, vocab_size, special_tokens: List[str] = None):
        special_tokens = special_tokens or []       # emtpy list if None
        self.vocab_size = vocab_size-len(special_tokens)        # leave some space for special tokens
        self.vocab: Dict[int, bytes] = {}
        self.merges: Dict[Tuple[int, int], int] = {}    # list of all the merges

        # GPT-4 regex
        self.reg = re.compile(r"""'(?i:[sdmtlre]|ll|ve)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
        self.vocab_special_tokens: Dict[str, int] = {}

        for i in range(len(special_tokens)):
            self.vocab_special_tokens[special_tokens[i]] = i + self.vocab_size

    @staticmethod
    def replace_pair(list_, pair, value):
        updated_list = list()
        i = 0
        while i<len(list_):
            if i < len(list_) - 1 and list_[i]==pair[0] and list_[i + 1]==pair[1]:
                updated_list.append(value)
                i+=2
            else:
                updated_list.append(list_[i])
                i+=1
        return updated_list


    def train(self, data, verbose=False):
        if self.vocab_size<256: raise ValueError("Vocabulary size must be at least 256 to accommodate raw bytes.")
        num_merges = self.vocab_size - 256
        self.vocab = {i: bytes([i]) for i in range(256)}  # initially, all bytes
        for token, special_id in self.vocab_special_tokens.items():     # then add special tokens
            self.vocab[special_id] = token.encode('utf-8')

        print("Applying regex pre-tokenization split...")
        chunks = [list(chunk.encode('utf-8')) for chunk in self.reg.findall(data)]
        del data

        print(f"Starting BPE training for {num_merges} merges...")

        # find pairs number of occurrences once, we'll update this with every merge, removing O(n^2) bottleneck
        pairs = {}
        pair_to_chunks = collections.defaultdict(set)   # maps every pair tuple -> a set of chunk indices that contain it

        for chunk_idx, chunk in enumerate(chunks):
            for pair in zip(chunk, chunk[1:]):
                pairs[pair] = pairs.get(pair, 0) + 1
                pair_to_chunks[pair].add(chunk_idx)

        for i in range(num_merges):
            if not pairs:
                break  # Nothing left to merge

            best_pair = max(pairs, key=pairs.get)

            # replace anywhere the pair occurred with their new encoding(256+i)
            new_id = 256+i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # apply merge across only affected chunks evading O(n^2)
            affected_chunk_indices = list(pair_to_chunks.get(best_pair, set()))
            for chunk_idx in affected_chunk_indices:
                old_chunk = chunks[chunk_idx]
                # step 1 => remove all current pairs of this specific chunk from pairs temporarily
                for pair in zip(old_chunk, old_chunk[1:]):
                    pairs[pair] -= 1
                    if pairs[pair] <= 0:
                        del pairs[pair]

                    pair_to_chunks[pair].discard(chunk_idx)     # well the pair is not in this chunk anymore because we just removed it
                    if not pair_to_chunks[pair]:
                        del pair_to_chunks[pair]

                # Step 2 => Merge the pair inside this single isolated chunk
                new_chunk = self.replace_pair(old_chunk, best_pair, new_id)
                chunks[chunk_idx] = new_chunk

                # Step 3 => Register the newly formed pairs from this chunk into global tracking
                for pair in zip(new_chunk, new_chunk[1:]):
                    pairs[pair] = pairs.get(pair, 0) + 1
                    pair_to_chunks[pair].add(chunk_idx)

            if verbose and i % 10 == 0:
                print(f"Merge {i + 1}/{num_merges}: {best_pair} -> {new_id} ({self.vocab[new_id].decode('utf-8', errors='replace')!r})")



    def _encode(self, text) -> List[int]:
        """Converts a raw string into a list of token IDs using learned merge rules.
        doesn't handle special tokens """

        text_chunks = self.reg.findall(text)
        final_ids = []

        # We process each chunk separately
        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))

            while len(chunk_ids) >= 2:     # obviously we cant merge for smaller strings
                # find all valid merge candidates in this chunk that exist in our vocabulary
                valid_pairs = [p for p in zip(chunk_ids, chunk_ids[1:]) if p in self.merges]
                if not valid_pairs: # no more eligible merge transformations available
                    break

                # find the pair that was merged earliest during the training cycle
                pair = min(valid_pairs, key=lambda p: self.merges[p])

                new_id = self.merges[pair]      # new id assigned for that pair
                chunk_ids = self.replace_pair(chunk_ids, pair, new_id)
            final_ids.extend(chunk_ids)

        return final_ids


    def encode(self, text, special_tokens_allowed: bool=False) -> List[int]:
        if not self.vocab_special_tokens:   # no special tokens anyway
            return self._encode(text)
        if not special_tokens_allowed:
            return self._encode(text)

        special_pattern = re.compile("(" + "|".join(re.escape(t) for t in self.vocab_special_tokens.keys()) + ")")
        parts = special_pattern.split(text)

        final_ids = []
        for part in parts:      # in the chopped of pieces, if the piece is a special token, use special token's table. else do normal
            if part in self.vocab_special_tokens:
                # Directly map the special token to its ID
                final_ids.append(self.vocab_special_tokens[part])
            elif part:
                # Regular text segment gets processed by normal BPE
                final_ids.extend(self._encode(part))

        return final_ids



    def decode(self, list_ids: List[int]) -> str:
        text_bytes = b"".join(self.vocab[idx] for idx in list_ids)
        return text_bytes.decode("utf-8", errors="replace")

    def save(self, file_path: str):
        """Saves the tokenizer state to disk."""
        state = {
            "vocab": self.vocab,
            "merges": self.merges,
            "vocab_size": self.vocab_size,
            "vocab_special_tokens": self.vocab_special_tokens
        }
        with open(file_path, "wb") as f:
            pickle.dump(state, f)
        print(f"Tokenizer saved to {file_path}")

    def load(self, file_path: str):
        """Loads the tokenizer state from disk."""
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        self.vocab = state["vocab"]
        self.merges = state["merges"]
        self.vocab_size = state["vocab_size"]
        self.vocab_special_tokens = state.get("vocab_special_tokens", {})
        print(f"Tokenizer loaded from {file_path}")

