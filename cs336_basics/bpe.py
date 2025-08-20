import os
import regex as re
import copy

ENCODE = "utf-8"

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # init vocab, mergelist
    merge_list = []
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        token_bytes = token.encode(encoding=ENCODE)
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes

    with open(input_path) as f:
        raw_text = f.read()

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
    
    return (vocab, merge_list)
