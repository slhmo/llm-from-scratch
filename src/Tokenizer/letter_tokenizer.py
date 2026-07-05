import torch



class Tokenizer:
    def __init__(self):
        self.all_tokens = None
        self.char_to_int=None
        self.int_to_char=None
        self.vocab_size = None
        self.data = list()

    def tokenize(self, paths_to_data):
        chars = list()
        for path in paths_to_data:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                self.data.extend(text)
                chars.extend(list(set(text)))

        chars.sort()
        self.vocab_size = len(chars)
        self.char_to_int = {ch: i for i, ch in enumerate(chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(chars)}
        self.all_tokens = torch.tensor(self.encode(self.data), dtype=torch.long)

    def encode(self, String_):
        return [self.char_to_int[c] for c in String_]

    def decode(self, Integers):
        return ''.join([self.int_to_char[i] for i in Integers])


    # def has_next(self, step, batch_size):
    #     return step*batch_size+batch_size<len(self.data)
    #
    # def get_next_batch(self, batch_size, context_window):
    #     # Pull random starting indices across the entire text
    #     ix = torch.randint(0, len(self.all_tokens) - context_window - 1, (batch_size,))
    #     x = torch.stack([self.all_tokens[i: i + context_window] for i in ix])
    #     y = torch.stack([self.all_tokens[i + 1: i + context_window + 1] for i in ix])
    #     return x.to(DEVICE), y.to(DEVICE)

