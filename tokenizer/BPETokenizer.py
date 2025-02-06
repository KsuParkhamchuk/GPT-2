import regex as re
from hyperparams import GPT4_SPLIT_PATTERN

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
        self.pattern = GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.merges = {}

    # get all possibile byte pairs from a sequence
    def get_pairs(self, bytes_list, counted_pairs=None):
        counted_pairs = {} if counted_pairs is None else counted_pairs

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
        n_merges = self.vocab_size - 259
        sequence_chunks = re.findall(self.compiled_pattern, sequence)
        merges = {}
        encoded_chunks = [list(ch.encode("utf-8")) for ch in sequence_chunks]

        for i in range(n_merges):
            pairs = {}

            for ch in encoded_chunks:
                self.get_pairs(ch, pairs)

            frequent_pair = max(pairs, key=pairs.get)
            pair_id = 260 + i
            self.update_vocabulary(frequent_pair)
            merges[frequent_pair] = pair_id
            encoded_chunks = [
                self.merge(ch, frequent_pair, pair_id) for ch in encoded_chunks
            ]
            print(
                f"merge {i+1}/{n_merges}: {frequent_pair} -> {pair_id} ({self.id_to_token[pair_id]}) had {pairs[frequent_pair]} occurrences"
            )
        self.merges = merges

    def encode_chunk(self, chunk):
        ids = list(chunk)
        while len(ids) >= 2:
            pairs = self.get_pairs(ids)
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            ids = self.merge(ids, pair, self.merges[pair])

        return ids

    def encode(self, sequence):
        # Convert to list of byte token IDs (0-255 initially)
        sequence_chunks = re.findall(self.compiled_pattern, sequence)
        encoded_chunks = [ch.encode("utf-8") for ch in sequence_chunks]
        tokens = []
        for ch in encoded_chunks:
            tokens.extend(self.encode_chunk(ch))

        padded_tokens = self.pad_sequence(tokens=tokens)

        return padded_tokens

    def decode(self, tokens):
        return b"".join(self.id_to_token.get(token) for token in tokens).decode(
            "utf-8", errors="replace"
        )
