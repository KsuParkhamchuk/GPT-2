from hyperparams import VOCABULARY_SIZE

sequence = "The join() method in Python is used to concatenate the elements of an iterable (such as a list, tuple, or set) into a single string with a specified delimiter placed between each element. Lets take a simple example to join list of string using join() method."


class BPETokenizer:
    def __init__(self, vocab_size, context_size):
        self.special_tokens = [b"<PAD>", b"<UNK>", b"<BOS>", b"<EOS>"]
        # Create two-way mappings
        self.token_to_id = {bytes([i]): i for i in range(256)}
        self.id_to_token = {i: bytes([i]) for i in range(256)}
        # Add special tokens
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = 256 + i
            self.id_to_token[256 + i] = token
        self.next_token_id = len(self.token_to_id)
        self.context_size = context_size
        self.vocab_size = vocab_size

    # get all possibile byte pairs from a sequence
    def get_pairs(self, bytes_list):
        counted_pairs = {}

        for pair in zip(bytes_list, bytes_list[1:]):
            counted_pairs[pair] = counted_pairs.get(pair, 0) + 1

        return counted_pairs

    # convert token to its byte representation
    def get_bytes(self, value):
        # Direct lookup instead of searching
        return self.id_to_token.get(value, b"<UNK>")

    def update_vocabulary(self, new_pair):
        # Get byte representations from IDs
        first_byte = self.id_to_token[new_pair[0]]
        second_byte = self.id_to_token[new_pair[1]]
        new_token = first_byte + second_byte

        # Add to both mappings
        self.token_to_id[new_token] = self.next_token_id
        self.id_to_token[self.next_token_id] = new_token
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
        pad_id = self.token_to_id[b"<PAD>"]
        return tokens + [pad_id] * n_pads

    def clean_sequence(self, sequence):
        pass

    def train(self, sequence):
        n_merges = self.vocab_size - 256
        i = 0
        bytes_list = sequence.encode("utf-8")

        while i < n_merges:
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
            print(pairs)
            if not pairs:
                break

            # Find the first mergeable pair with lowest token ID
            min_pair = None
            min_id = float("inf")
            for pair in pairs:
                # Convert both tokens in pair to their byte representations
                byte_pair = self.get_bytes(pair[0]) + self.get_bytes(pair[1])
                token_id = self.token_to_id.get(byte_pair, None)
                if token_id is not None and token_id < min_id:
                    min_id = token_id
                    min_pair = pair

            if min_pair is None:
                break

            tokens = self.merge(tokens, min_pair, min_id)

        padded_tokens = self.pad_sequence(tokens=tokens)

        return padded_tokens

    def decode(self, tokens):
        return b"".join(self.id_to_token.get(token) for token in tokens).decode(
            "utf-8", errors="replace"
        )
