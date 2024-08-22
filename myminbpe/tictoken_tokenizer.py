import tiktoken

from myminbpe.regex_tokenizer import RegexTokenizer


def recover_merges(mergeable_ranks):
    # the `merges` are already the byte sequences in their merged state.
    # so we have to recover the original pairings. We can do this by doing
    # a small BPE training run on all the tokens, in their order.
    # also see https://github.com/openai/tiktoken/issues/60
    # also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue  # skip raw bytes
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges


def bpe(mergeable_ranks, token, max_rank):
    # helper function used in get_gpt4_merges() to reconstruct the merge forest
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


class G4Tokenizer(RegexTokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        self.byte_shuffle = {i: enc._mergeable_ranks[bytes([i])] for i in range(self.bytes_size)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        self.merges = recover_merges(mergeable_ranks)
        self.vocabs = {}
        for i in range(self.bytes_size):
            self.vocabs[i] = bytes([i])
        for (id0, id1), new_id in self.merges.items():
            self.vocabs[new_id] = self.vocabs[id0] + self.vocabs[id1]

    def encode(self, text):
        _bytes = list(text.encode(encoding='utf-8'))
        _bytes = [self.byte_shuffle[b] for b in _bytes]
        while len(_bytes) > 2:
            counts = self._get_stats(_bytes)
            pair = self._get_top_pair(counts)
            new_id = self.merges.get(pair)
            if new_id is None:
                break
            _bytes = self._merge(_bytes, pair, new_id)
        return _bytes

    def decode(self, ids):
        tokens = b''.join(self.vocabs[_id] for _id in ids)
        tokens = bytes(self.inverse_byte_shuffle[b] for b in tokens)
        return tokens.decode(encoding='utf-8')


_tokenizer = G4Tokenizer()
text = "hello world!!!? (안녕하세요!) lol123 😉"
ids = _tokenizer.encode(text)
tokenizer_decoded = _tokenizer.decode(ids)
print(tokenizer_decoded)

enc = tiktoken.get_encoding("cl100k_base") # this is the GPT-4 tokenizer
ids = enc.encode(text)
tictoken_decoded = enc.decode(ids) # get the same text back
print(tictoken_decoded)

print(tokenizer_decoded == tictoken_decoded)