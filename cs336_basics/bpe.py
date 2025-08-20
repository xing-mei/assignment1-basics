import os
import regex as re
import copy
import time
from concurrent.futures import ProcessPoolExecutor
from cs336_basics.pretokenization_example import find_chunk_boundaries

ENCODE = "utf-8"

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
        print("path 1")
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
        print("path 2")
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

    print(f'file loading + pretokenization: {t2 - t1}')
    print(f'merging: {t3 - t2}')
    
    return (vocab, merge_list)

if __name__ == "__main__":
    # Problem (train_bpe_tinystories)
    vocab_1, merge_1 = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ['<|endoftext|>'], True)
    vocab_2, merge_2 = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ['<|endoftext|>'])
    assert set(vocab_1.keys()) == set(vocab_2.keys())
    assert set(vocab_1.values()) == set(vocab_2.values())
    assert merge_1 == merge_2

    # Problem (train_bpe_expts_owt)
