from hyperparams import N_MERGES

sequence = "The join() method in Python is used to concatenate the elements of an iterable (such as a list, tuple, or set) into a single string with a specified delimiter placed between each element. Lets take a simple example to join list of string using join() method."


class BPETokenizer:
    def __init__(self, context_size):
        self.special_tokens = [b"<PAD>", b"<UNK>", b"<BOS>", b"<EOS>"]
        # init base vocabulary
        self.vocab = {bytes([i]): i for i in range(256)}
        # update with special tokens
        self.vocab.update(
            {token: i + 256 for i, token in enumerate(self.special_tokens)}
        )
        self.next_token_id = len(self.vocab)
        self.context_size = context_size

    # get all possibile byte pairs from a sequence
    def get_pairs(self, bytes_list):
        counted_pairs = {}

        for pair in zip(bytes_list, bytes_list[1:]):
            counted_pairs[pair] = counted_pairs.get(pair, 0) + 1

        return counted_pairs

    # convert token to its byte representation
    def get_bytes(self, value):
        if value <= 255:
            return bytes([value])
        # If value is a token ID, find its bytes representation in vocab
        for token_bytes, token_id in self.vocab.items():
            if token_id == value:
                return token_bytes
        raise ValueError(f"Token ID {value} not found in vocabulary")

    def update_vocabulary(self, new_pair):
        new_token = self.get_bytes(new_pair[0]) + self.get_bytes(new_pair[1])
        self.vocab[new_token] = self.next_token_id
        self.next_token_id += 1

    def merge(self, bytes_list, frequent_pair, replacement):
        i = 0
        updated_sequence = []
        while i < len(bytes_list):
            if (
                i + 1 < len(bytes_list)
                and (bytes_list[i], bytes_list[i + 1]) == frequent_pair
            ):
                updated_sequence.append(replacement)
                i += 2
            else:
                updated_sequence.append(bytes_list[i])
                i += 1

        return updated_sequence

    def pad_sequence(self, tokens):
        n_pads = self.context_size - len(tokens)
        return tokens + [self.vocab[b"<PAD>"]] * n_pads

    def clean_sequence(self, sequence):
        pass

    def train(self, sequence):
        i = 0
        bytes_list = sequence.encode("utf-8")

        while i < N_MERGES:
            pairs = self.get_pairs(bytes_list)
            frequent_pair = max(pairs, key=pairs.get)
            self.update_vocabulary(frequent_pair)
            bytes_list = self.merge(bytes_list, frequent_pair, self.next_token_id - 1)
            i += 1

    def encode(self, sequence):
        # Convert to list of byte token IDs (0-255 initially)
        tokens = list(sequence.encode("utf-8"))

        # Keep merging until no more merges possible
        while True:
            pairs = self.get_pairs(tokens)
            if not pairs:
                break

            # Find the first mergeable pair with lowest token ID
            min_pair = None
            min_id = float("inf")
            for pair in pairs:
                # Convert both tokens in pair to their byte representations
                byte_pair = self.get_bytes(pair[0]) + self.get_bytes(pair[1])
                token_id = self.vocab.get(byte_pair, None)
                if token_id is not None and token_id < min_id:
                    min_id = token_id
                    min_pair = pair

            if min_pair is None:
                break

            tokens = self.merge(tokens, min_pair, min_id)
        return tokens

    def decode(self, tokens):
        return b"".join(self.get_bytes(token) for token in tokens).decode(
            "utf-8", errors="replace"
        )

    def forward(self, sequence):
        tokens = self.encode(sequence=sequence)
        padded_tokens = self.pad_sequence(tokens=tokens)

        return padded_tokens


# tokenizer = BPETokenizer(30)
# tokenizer.train(sequence=sequence)
# print(list(tokenizer.encode("I am swimming")))
# print(tokenizer.decode(tokenizer.encode("I am swimming")))
# print(tokenizer.forward("I am swimming"))
