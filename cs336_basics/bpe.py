import os
import copy
import time
import json
import regex as re
from typing import BinaryIO, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor

ENCODE = "utf-8"

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_file_chunk(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str], 
) -> dict[tuple[bytes], int]:
    
    with open(input_path, "rb") as f:
        f.seek(start)
        raw_text = f.read(end - start).decode(ENCODE, errors="ignore")

    # split text with special tokens
    split_pattern = "|".join([re.escape(st) for st in special_tokens])
    splitted_texts = [text for text in re.split(split_pattern, raw_text) if text]

    # pretokenize each splitted text and put all tokens into a dict
    token_dict = {}
    token_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for splitted_text in splitted_texts:
        for match in re.finditer(token_pattern, splitted_text):
            token_key = tuple([bytes([c]) for c in match.group(0).encode(ENCODE)])
            token_dict[token_key] = token_dict.get(token_key, 0) + 1
        
    return token_dict

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    use_multiprocess: bool = False, 
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # init vocab, mergelist
    merge_list = []
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        token_bytes = token.encode(encoding=ENCODE)
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes

    t1 = time.time()
    if use_multiprocess and len(special_tokens) == 1 and special_tokens[0] == "<|endoftext|>":
        # use multiprocess only in this condition
        token_dict = {}
        num_process = 4
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_process, b"<|endoftext|>")
        futures = []
        with ProcessPoolExecutor(max_workers = num_process) as executor:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                futures.append(executor.submit(process_file_chunk, input_path, start, end, special_tokens))
        for future in futures:
            # block get
            local_token_dict = future.result() 
            for k, v in local_token_dict.items():
               token_dict[k] = token_dict.get(k, 0) + v
    else:
        with open(input_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
        token_dict = process_file_chunk(input_path, 0, file_size, special_tokens)
    t2 = time.time()

    # find pairs with max counts
    # use pair_token_mapping for fast pair->token retrieval

    pair_count = {}
    pair_token_mapping = {}
    for token, count in token_dict.items():
        for pair in zip(token, token[1:]):
            pair_count[pair] = pair_count.get(pair, 0) + count
            if pair in pair_token_mapping:
                pair_token_mapping[pair].add(token)
            else:
                pair_token_mapping[pair] = {token}

    # merge pairs, update merge list, vocab
    while(len(vocab) < vocab_size):
        pair_with_max_counts = max(pair_count, key=lambda x: (pair_count.get(x), x))
        
        # update merge list, vocab 
        merged_id = len(vocab)
        merged_bytes = pair_with_max_counts[0] + pair_with_max_counts[1]
        merge_list.append(pair_with_max_counts)
        vocab[merged_id] = merged_bytes
        
        # update token dict, pair_count, pair_token_mapping
        update_token_set = copy.deepcopy(pair_token_mapping[pair_with_max_counts])
        for token in update_token_set:
            count = token_dict[token]
            new_token = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and (token[i], token[i + 1]) == pair_with_max_counts:
                    new_token.append(merged_bytes)
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            new_token = tuple(new_token)
            del token_dict[token]
            token_dict[new_token] = count

            for pair in zip(token, token[1:]):
                pair_count[pair] -= count
                pair_token_mapping[pair].discard(token)

            for pair in zip(new_token, new_token[1:]):
                pair_count[pair] = pair_count.get(pair, 0) + count
                if pair in pair_token_mapping:
                    pair_token_mapping[pair].add(new_token)
                else:
                    pair_token_mapping[pair] = {new_token}

        del pair_count[pair_with_max_counts]
    t3 = time.time()

    print(f'\nfile loading + pretokenization: {t2 - t1}')
    print(f'merging: {t3 - t2}')
    
    return (vocab, merge_list)

class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = None
        self.special_tokens_set = None
        self.split_pattern = None
        if special_tokens:
            # sort to solve special token overlapping issues in the text
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_tokens_set = set(self.special_tokens)
            pattern = "|".join([re.escape(st) for st in self.special_tokens])
            self.split_pattern = "(" + pattern + ")"
    
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, encoding="utf-8") as f:
            inv_vocab = json.load(f)
            vocab = {id: token.encode(ENCODE) for token, id in inv_vocab.items()}

        with open(merges_filepath, encoding="utf-8") as f:
            merges = [tuple(line.rstrip().split(" ")) for line in f]
            merges_bytes = [(token_1.encode(ENCODE), token_2.encode(ENCODE)) for token_1, token_2 in merges]
        
        return cls(vocab, merges_bytes, special_tokens)

    def encode(
        self,
        text: str,
    ) -> list[int]:
        
        # split text with special tokens (included)
        split_texts = [text]
        if self.split_pattern:
            split_texts = [split_text for split_text in re.split(self.split_pattern, text) if split_text]

        # tokenize each token
        token_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""        
        token_ids = []
        for split_text in split_texts:
            if self.special_tokens_set and split_text in self.special_tokens_set:
                token_ids.append(self.inv_vocab[split_text.encode(ENCODE)])
            else:
                for match in re.finditer(token_pattern, split_text):
                    token_bytes = tuple([bytes([c]) for c in match.group(0).encode(ENCODE)])
                    # merge bytes by order
                    for pair in self.merges:
                        i = 0
                        new_token_bytes = []
                        while i < len(token_bytes):
                            if i < len(token_bytes) - 1 and (token_bytes[i], token_bytes[i + 1]) == pair:
                                new_token_bytes.append(token_bytes[i] + token_bytes[i + 1])
                                i += 2
                            else:
                                new_token_bytes.append(token_bytes[i])
                                i += 1
                        token_bytes = tuple(new_token_bytes)
                    for final_bytes in token_bytes:
                        token_ids.append(self.inv_vocab[final_bytes])
        return token_ids
    
    def encode_iterable(
        self, 
        iterable: Iterable[str],
    ) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(
        self, 
        ids: list[int],
    ) -> str:
        output_bytes = bytes()
        for id in ids:
            output_bytes = output_bytes + self.vocab[id]
        return output_bytes.decode(ENCODE, errors='replace')

if __name__ == "__main__":
    # Problem (train_bpe_tinystories)
    vocab_1, merge_1 = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ['<|endoftext|>'], True)
    vocab_2, merge_2 = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ['<|endoftext|>'])
    assert set(vocab_1.keys()) == set(vocab_2.keys())
    assert set(vocab_1.values()) == set(vocab_2.values())
    assert merge_1 == merge_2

    # Problem (train_bpe_expts_owt)
