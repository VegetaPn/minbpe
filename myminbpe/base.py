# -*- coding: utf-8 -*-

class MyTokenizer:
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def _get_top_pair(self, counts):
        return max(counts, key=counts.get)

    def _get_stats(self, bytes):
        counts = {}
        for b0, b1 in zip(bytes[:], bytes[1:]):
            count = counts.get((b0, b1), 0)
            counts[(b0, b1)] = count + 1
        return counts

    def _merge(self, bytes, top_pair, new_id):
        new_bytes = []
        i = 0
        while i < len(bytes):
            if i < len(bytes) - 1 and bytes[i] == top_pair[0] and bytes[i+1] == top_pair[1]:
                new_bytes.append(new_id)
                i += 2
            else:
                new_bytes.append(bytes[i])
                i += 1
        return new_bytes