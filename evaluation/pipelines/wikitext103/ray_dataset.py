import pathlib
import torch
import torchtext.transforms as T
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import ray
import time

DATASET_LOC = "datasets/wikitext103"
DATASET_NAME = "wikitext103"
DATASET_FILE = "wikitext-103-v1.zip"
DATASET_SOURCE = "https://s3.amazonaws.com/research.metamind.io/wikitext/\
wikitext-103-v1.zip"
WARMUP_SAMPLES = 50000


class Tokenizer:
    def __init__(self):
        encoder_json_path = (
            "https://download.pytorch.org/models/text/gpt2_bpe_encoder.json"
        )
        vocab_bpe_path = (
            "https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe"
        )
        self.tokenizer = T.GPT2BPETokenizer(encoder_json_path, vocab_bpe_path)

    def __call__(self, x):
        return {"text": self.tokenizer(x["text"])}


class Vocab:
    def __init__(self):
        vocab_path = (
            "https://download.pytorch.org/models/text/roberta.vocab.pt"
        )
        self.vocab = T.VocabTransform(load_state_dict_from_url(vocab_path))

    def __call__(self, x):
        return {"text": self.vocab(x["text"])}


class Embedding:
    def __init__(self):
        self.embedding = nn.Embedding(50257, 764, _freeze=True)

    def __call__(self, x):
        return {"text": self.embedding(torch.as_tensor(x["text"]))}


def tokenize(x):
    encoder_json_path = (
        "https://download.pytorch.org/models/text/gpt2_bpe_encoder.json"
    )
    vocab_bpe_path = (
        "https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe"
    )
    tokenizer = T.GPT2BPETokenizer(encoder_json_path, vocab_bpe_path)
    return {"text": tokenizer(x["text"])}


def add_eos(x):
    f = T.AddToken(token=0, begin=True)
    return {"text": f(x["text"])}


def add_bos(x):
    f = T.AddToken(token=2, begin=False)
    return {"text": f(x["text"])}


def to_tensor(x):
    f = T.ToTensor()
    return {"text": f(x["text"])}


class Timer:
    def __init__(self):
        self._start = None
        self._end = None

    def __enter__(self):
        self._start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.perf_counter()

    def reset(self):
        self._start = time.perf_counter()

    def delta(self):
        if self._start is None or self._end is None:
            raise RuntimeError()
        return self._end - self._start


def build_ds(root):
    # Get list of dirs
    ds = ray.data.read_text(str(root))
    ds = ds.map(
        fn=Tokenizer,
        compute=ray.data.ActorPoolStrategy(min_size=1, max_size=32),
    )
    ds = ds.map(
        fn=Vocab,
        compute=ray.data.ActorPoolStrategy(min_size=1, max_size=32),
    )
    ds = ds.map(add_eos)
    ds = ds.map(add_bos)
    ds = ds.map(
        fn=Embedding,
        compute=ray.data.ActorPoolStrategy(min_size=1, max_size=32),
    )

    return ds


def get_dataset():
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    train_filepath = pathlib.Path(data_dir) / pathlib.Path(
        "wikitext-103/wiki.train.tokens"
    )
    ds = build_ds(str(train_filepath))
    ray.data.DataContext.get_current().execution_options.locality_with_output = (
        True
    )
    return ds


if __name__ == "__main__":
    ds = get_dataset()

    # for row in ds.iter_rows():
    #     print(row)
    #     break
    timer = Timer()
    with timer:
        for idx, row in enumerate(ds.iter_rows()):
            if idx == WARMUP_SAMPLES:
                timer.reset()
            if idx == WARMUP_SAMPLES + 100000:
                break

            if idx % 10000 == 0:
                print(idx)

    print(timer.delta())
