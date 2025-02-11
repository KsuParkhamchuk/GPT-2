import regex as re
from collections import Counter
from hyperparams import GPT4_SPLIT_PATTERN, BASE_VOCABULARY_SIZE, N_RAW_BYTES
import json
from typing import List, Tuple


class BPETokenizer:
    def __init__(self, vocab_size, context_size):
        self.special_tokens = [b"<PAD>", b"<UNK>"]
        # Create two-way mappings
        self.token_to_id = {bytes([i]): i for i in range(N_RAW_BYTES)}
        self.id_to_token = {i: bytes([i]) for i in range(N_RAW_BYTES)}
        # Add special tokens
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = N_RAW_BYTES + i
            self.id_to_token[N_RAW_BYTES + i] = token

        self.next_token_id = len(self.token_to_id)
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.pattern = GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.merges = {}
        self.unknown_token_id = 257
        self.pad_token_id = 256

    # get all possibile byte pairs from a sequence
    def get_pairs(
        self, bytes_list: List[int], counted_pairs=None
    ) -> Counter[Tuple[int, int]]:
        """
        Counts pairs in aech chunk of sequence and updates external variable pairs

        parameters:
            bytes_list - list of tokens [int, int, ...]
            counted_pairs - might be None when first iteration, otherwise - Counter object of counted turples Counter((int, int): int), (), () ...)

        Uses zip to create pairs
        Uses Counter from collections to update the pairs counter
        """
        counted_pairs = Counter() if counted_pairs is None else counted_pairs
        counted_pairs.update(Counter(zip(bytes_list, bytes_list[1:])))

        return counted_pairs

    # convert token to its byte representation
    def get_bytes(self, value: int) -> bytes:
        return self.id_to_token.get(value, self.id_to_token[self.unknown_token_id])

    def update_vocabulary(self, new_pair: Tuple[int, int]):
        """
        Update vocabulary while training by merging pair of bytes into
        byte object and assigning the next available id

        for faster access:
        token_to_id - dictionary {byte_str: int}
        id_to_token - dictionary {int: byte_str}

        1. Search for the bite representation in id_to_token
        2. Merge two token by concatenating them
        3. Update both token_to_id and id_to_token objects
        4. Increment next_token_id value by one for the next update
        """
        # Get byte representations from IDs
        first_byte = self.id_to_token[new_pair[0]]
        second_byte = self.id_to_token[new_pair[1]]
        new_token = first_byte + second_byte

        # Add to both mappings
        token_id = self.next_token_id
        self.token_to_id[new_token] = token_id
        self.id_to_token[token_id] = new_token
        self.next_token_id += 1

    def merge(
        self, bytes_list: List[int], frequent_pair: Tuple[int], replacement: int
    ) -> List[int]:
        """
        Merges the most frequent pair in a chunk of bytes

        If charachter is not the last in a sequence and (curr_ch, next_ch) equals to the most frequent pair:
            - replce and skip 2 positions
        otherwise:
            - add byte to updated sequence as it is and move to the next
        """
        i = 0
        updated_sequence = []
        n = len(bytes_list)

        while i < n:
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

    def pad_sequence(self, tokens: List[int]) -> List[int]:
        """
        Add n number of special tokens to the end of the sequence to fill the context size
        """
        n_pads = self.context_size - len(tokens)
        return tokens + [self.pad_token_id] * n_pads

    def process_special_tokens(self, tokens: List[int]) -> List[int]:
        """
        Process special tokens <UNK> or <PAD> while decoding

        1. Removes <PAD> tokens by skiping them
        2. Append <UNK> token if the processed token is not in vocabulary
        """
        processed_tokens = []
        print(self.id_to_token)
        for token in tokens:
            if token == self.pad_token_id:
                continue
            elif token in self.id_to_token:
                processed_tokens.append(token)
            else:
                print(token)
                processed_tokens.append(self.unknown_token_id)

        return processed_tokens

    def train(self, sequence: str):
        """Fill tokenizer vocabulary with n_merges"""
        n_merges = self.vocab_size - BASE_VOCABULARY_SIZE
        sequence_chunks = re.findall(self.compiled_pattern, sequence)
        merges = {}
        encoded_chunks = [list(ch.encode("utf-8")) for ch in sequence_chunks]

        for i in range(n_merges):
            pairs = Counter()

            for ch in encoded_chunks:
                self.get_pairs(ch, pairs)

            frequent_pair = max(pairs, key=pairs.get)
            pair_id = BASE_VOCABULARY_SIZE + i
            self.update_vocabulary(frequent_pair)
            merges[frequent_pair] = pair_id
            encoded_chunks = [
                self.merge(ch, frequent_pair, pair_id) for ch in encoded_chunks
            ]
            print(
                f"merge {i+1}/{n_merges}: {frequent_pair} -> {pair_id} ({self.id_to_token[pair_id]}) had {pairs[frequent_pair]} occurrences"
            )
        self.merges = merges
        self.save_vocab("vocab.json")

    def encode_chunk(self, chunk: bytes) -> List[int]:
        """
        Transforms one chunk of initial sequence into tokens

        1. Form byte pairs
        2. Find the priority merge among pairs of chunk (the min value = the most early merge)
        3. Replace found pair with a token id
        4. Break if found pair is not in merges
        """
        ids = list(chunk)
        while len(ids) >= 2:
            pairs = self.get_pairs(ids)
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            ids = self.merge(ids, pair, self.merges[pair])

        return ids

    def encode(self, sequence: str) -> List[int]:
        """
        Transforms the whole sequence into tokens by splitting it into chunks and process them separately

        1. Split into chunks with a help of regex (separating punctuation, numbers, etc.)
        2. Encode each chunk with UTF-8
        3. Convert each chunk to a sequence of tokens
        4. Extend token sequence with n special token <PAD> - 256 to fill context_size
        """
        # Convert to list of byte token IDs (0-255 initially)
        sequence_chunks = re.findall(self.compiled_pattern, sequence)
        encoded_chunks = [ch.encode("utf-8") for ch in sequence_chunks]
        tokens = []
        for ch in encoded_chunks:
            tokens.extend(self.encode_chunk(ch))

        padded_tokens = self.pad_sequence(tokens=tokens)

        return padded_tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decode tokens back to text

        1. Process special tokens (replace unknown tokens and skip pad)
        2. Decode each token
        3. Raise an Exception if sequence is empty
        """
        if len(tokens) == 0:
            raise "Sorry, the generation error occurs"

        tokens = self.process_special_tokens(tokens)

        return b"".join(self.id_to_token[token] for token in tokens).decode(
            "utf-8", errors="replace"
        )

    def save_vocab(self, file_path: str):

        vocab_data = {
            "merges": {f"{a},{b}": idx for (a, b), idx in self.merges.items()},
            "id_to_token": {k: list(v) for k, v in self.id_to_token.items()},
            "special_tokens": [list(t) for t in self.special_tokens],
            "vocab_size": self.vocab_size,
            "context_size": self.context_size,
            "pattern": self.pattern,
        }

        with open(file_path, "w") as f:
            json.dump(vocab_data, f, indent=2)
