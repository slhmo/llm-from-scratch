import pickle
from typing import List, Dict, Tuple        # increase readability
from collections import Counter
import regex as re

class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab: Dict[int, bytes] = {}
        self.merges: Dict[Tuple[int, int], int] = {}    # list of all the merges

        # The GPT-2 regex split pattern ignore for now
        # self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @staticmethod
    def __replace_most_frequent_pair(text, replacement="XX"):
        # all consecutive pairs
        pairs = [text[i:i + 2] for i in range(len(text) - 1)]

        # count occurrences and find the most common one
        # .most_common(1) returns [(pair, count)]
        counts = Counter(pairs)
        if not counts:
            return text

        most_common_pair, _ = counts.most_common(1)[0]

        # 3. Replace the pair
        return text.replace(most_common_pair, replacement)

    @staticmethod
    def __get_consecutive_pairs(list_):
        """Counts all consecutive pairs of integers in a single pass."""
        counts = {}
        for pair in zip(list_, list_[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

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


    def train(self, paths_to_data, verbose=False):
        if self.vocab_size<256: raise ValueError("Vocabulary size must be at least 256 to accommodate raw bytes.")
        num_merges = self.vocab_size - 256
        self.vocab = {i: bytes([i]) for i in range(256)}  # initially, all bytes

        print("Loading and converting dataset to raw bytes...")
        raw_bytes = bytearray()
        for path in paths_to_data:
            with open(path, 'rb') as f:     # read binary
                raw_bytes.extend(f.read())

        # convert bytearray to a list of integers and delete that
        list_ids = list(raw_bytes)
        del raw_bytes
        print(f"Starting BPE training for {num_merges} merges...")

        for i in range(num_merges):
            # find pair with most occurrences
            pairs = self.__get_consecutive_pairs(list_ids)
            if not pairs:
                break  # Nothing left to merge

            best_pair = max(pairs, key=pairs.get)
            # replace anywhere the pair occurred with their new encoding(256+i)
            new_id = 256+i
            list_ids = self.replace_pair(list_ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            if verbose and i % 10 == 0:
                print(f"Merge {i + 1}/{num_merges}: {best_pair} -> {new_id} ({self.vocab[new_id].decode('utf-8', errors='replace')!r})")

    def encode(self, text) -> List[int]:
        """Converts a raw string into a list of token IDs using learned merge rules."""
        list_ids = list(text.encode("utf-8"))
        while len(list_ids) >= 2:     # obviously we cant merge for smaller strings
            pairs = self.__get_consecutive_pairs(list_ids)
            # find the pair that was merged earliest during the training cycle
            pair = min(pairs.keys(), key=lambda p: self.merges.get(p, float('inf')))

            if pair not in self.merges:
                break  # no more eligible merge transformations available

            new_id = self.merges[pair]      # new id assigned for that pair
            list_ids = self.replace_pair(list_ids, pair, new_id)
        return list_ids

    def decode(self, list_ids: List[int]) -> str:
        text_bytes = b"".join(self.vocab[idx] for idx in list_ids)
        return text_bytes.decode("utf-8", errors="replace")

    def save(self, file_path: str):
        """Saves the tokenizer state to disk."""
        state = {
            "vocab": self.vocab,
            "merges": self.merges,
            "vocab_size": self.vocab_size
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
        print(f"Tokenizer loaded from {file_path}")

