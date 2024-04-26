from pathlib import Path
from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(self, model_path: str) -> None:
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)

    @property
    def n_words(self):
        return self._model.vocab_size()

    @property
    def bos_id(self):
        return self._model.bos_id()

    @property
    def eos_id(self):
        return self._model.eos_id()

    @property
    def pad_id(self):
        return self._model.eos_id()

    def encode(self, s: str, bos: bool = False):
        t = self._model.Encode(s)
        if bos is True:
            t = [self.bos_id, *t]
        return t

    def decode(self, t: list):
        return self._model.Decode(t)


if __name__ == "__main__":
    path = Path("./tokenizer/tokenizer.model")
    # path = Path("./rabbit/misc/tokenizer/tokenizer/tokenizer.model")
    tokenizer = Tokenizer(model_path=str(path))
    print(tokenizer.n_words)
    print(tokenizer.n_words, tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id)

    text = f""".
"""
    print(text)
    print(len(text.split()))

    t = tokenizer.encode(s=text)
    print(t, len(t))

    print(tokenizer.decode(t))
    # print(tokenizer.decode([tokenizer.bos_id] + t + [tokenizer.eos_id]))
