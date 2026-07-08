import pandas as pd

from src.Configs import TRAIN_TOKENIZER_PATHS, VOCAB_SIZE, SPECIAL_TOKENS
from src.Tokenizer.BPE_tokenizer import BPETokenizer

if __name__=='__main__':        # first pairs take a lot of time because there are a lot of affected_chunks but as we go forward execution takes less time
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

    data = []
    for csv_path in TRAIN_TOKENIZER_PATHS:
        chunk_iterator = pd.read_csv(csv_path, chunksize=2000)
        for i, chunk in enumerate(chunk_iterator):
            for text in chunk['text']:
                if pd.isna(text):
                    continue
                data.append(text)

    string_content = "".join(data)
    tokenizer.train(string_content, verbose=True)


    encoded = tokenizer.encode("hello world")
    print(tokenizer.decode(encoded))

    # save it to disk
    tokenizer.save("tokenizer.pkl")

    # later, load it back into a new instance
    new_tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    new_tokenizer.load("tokenizer.pkl")

    # Verify
    encoded = new_tokenizer.encode("hello world")
    print(new_tokenizer.decode(encoded))