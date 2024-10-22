import pathlib
import tensorflow as tf
from transformers import GPT2Tokenizer
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
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def __call__(self, x):
        data = tf.convert_to_tensor(x["text"])
        return {
            "text": self.tokenizer(str(data.numpy()), return_tensors="tf")[
                "input_ids"
            ]
        }


def _truncate(x):
    dim = tf.shape(x["text"])[1]
    slice_size = tf.minimum(dim, 254)
    return {"text": tf.slice(x["text"], [0, 0], [1, slice_size])}


class Embedding:
    def __init__(self):
        self.embedding = tf.Variable(
            tf.random.uniform([50257, 764], -1.0, 1.0)
        )

    def __call__(self, x):
        return {"text": tf.nn.embedding_lookup(self.embedding, x["text"])}


#
#
# def add_eos(x):
#     f = T.AddToken(token=0, begin=True)
#     return {"text": f(x["text"])}
#
#
# def add_bos(x):
#     f = T.AddToken(token=2, begin=False)
#     return {"text": f(x["text"])}
#
#
# def to_tensor(x):
#     return {"text": tf.convert_to_tensor(x["text"])}


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
    ds = ds.map(_truncate)
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
