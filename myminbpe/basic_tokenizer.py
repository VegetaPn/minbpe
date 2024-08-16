class BasicTokenizer:
    def __init__(self):
        super().__init__()
        self.bytes_size = 0
        self.counts = {}
        self.merges = {}  # (byte0, byte1) --> new_byte
        self.ids = []

    def train(self, text, vocab_size, verbose=False):
        bytes = list(text.encode(encoding='utf-8'))
        self.bytes_size = len(set(bytes))
        print(f'self.bytes_size = {self.bytes_size}')

        for i in range(vocab_size - self.bytes_size):
            self.counts = self.__get_stats(bytes)
            top_pair = self.__get_top_pair(self.counts)
            bytes = self.__merge(bytes, top_pair, self.bytes_size + i + 1)
        self.ids = bytes

    def encode(self, text):
        bytes = list(text.encode(encoding='utf-8'))


    def __get_top_pair(self, counts):
        return max(counts, key=counts.get)

    def __get_stats(self, bytes):
        counts = {}
        for b0, b1 in zip(bytes[:], bytes[1:]):
            count = counts.get((b0, b1), 0)
            counts[(b0, b1)] = count + 1
        return counts

    def __merge(self, bytes, top_pair, new_id):
        new_bytes = []
        i = 0
        while i < len(bytes) - 1:
            pair = (bytes[i], bytes[i+1])
            if pair == top_pair:
                new_bytes.append(new_id)
                self.merges[pair] = new_id
                i += 2
            else:
                new_bytes.append(bytes[i])
                i += 1
        return new_bytes


text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
basic_tokenizer = BasicTokenizer()
basic_tokenizer.train(text, 100)
