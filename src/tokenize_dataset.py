import os
import pandas as pd
import numpy as np
from src.Tokenizer.BPE_tokenizer import BPETokenizer
from src.Configs import TOKENIZER_PATH, VOCAB_SIZE, SPECIAL_TOKENS, \
    RAW_TRAIN_PATH, TOKENIZED_TRAIN_PATH


def prepare_data(csv_path, output_bin_path, chunk_size=2000):
    # Initialize with the exact same special token setup
    tokenizer = BPETokenizer(VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
    tokenizer.load(TOKENIZER_PATH)

    # Extract the auto-assigned ID for your special token
    eot_id = tokenizer.vocab_special_tokens["<|endoftext|>"]

    print(f"Tokenizing {csv_path} -> {output_bin_path}")

    with open(output_bin_path, 'wb') as f_out:
        chunk_iterator = pd.read_csv(csv_path, chunksize=chunk_size)
        total_stories = 0

        for i, chunk in enumerate(chunk_iterator):
            chunk_tokens = []

            for text in chunk['text']:
                if pd.isna(text):
                    continue

                # encode actual story content
                ids = tokenizer.encode(str(text))
                chunk_tokens.extend(ids)

                # append the delimiter token after the story finishes
                chunk_tokens.append(eot_id)

            # flush the combined array segment to disk
            if chunk_tokens:
                np_array = np.array(chunk_tokens, dtype=np.uint16)
                f_out.write(np_array.tobytes())

            total_stories += len(chunk)
            print(f"Batch {i + 1} processed. Total stories tokenized: {total_stories:,}")

    print(f"Successfully created {output_bin_path} with EOT boundaries!")


if __name__ == '__main__':
    train_csv = RAW_TRAIN_PATH  # Change to your actual train split path
    prepare_data(train_csv, TOKENIZED_TRAIN_PATH)
