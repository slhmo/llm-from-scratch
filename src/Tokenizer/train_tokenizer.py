from src.Configs import data_paths, VOCAB_SIZE
from src.Tokenizer.BPE_tokenizer import BPETokenizer

if __name__=='__main__':
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(data_paths, verbose=True)

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