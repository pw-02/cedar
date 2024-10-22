import tensorflow as tf
import pathlib

from transformers import GPT2Tokenizer

from evaluation.tf_utils import TFEvalSpec

DATASET_LOC = "datasets/wikitext103"


def _load_text(path):
    text = tf.io.read_file(path)
    return tf.data.Dataset.from_tensor_slices(tf.strings.split(text, "\n"))


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
embedding = tf.Variable(tf.random.uniform([50257, 764], -1.0, 1.0))


@tf.py_function(Tout=tf.int32)
def _tokenize(x):
    return tokenizer(str(x.numpy()), return_tensors="tf")["input_ids"]


def _truncate(x):
    dim = tf.shape(x)[1]
    slice_size = tf.minimum(dim, 254)
    x = tf.slice(x, [0, 0], [1, slice_size])
    return x


def _embedding(x):
    return tf.nn.embedding_lookup(embedding, x)


def build_dataset(path, spec):
    # ds = _load_text(path)
    ds = tf.data.TextLineDataset(path)

    ds = ds.map(
        lambda x: _tokenize(x), num_parallel_calls=spec.num_parallel_calls
    )
    ds = ds.map(_truncate, num_parallel_calls=spec.num_parallel_calls)
    ds = ds.map(_embedding, num_parallel_calls=spec.num_parallel_calls)

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    if spec.service_addr:
        print(
            "Using tf.data.service with address {}".format(spec.service_addr)
        )
        ds = ds.apply(
            tf.data.experimental.service.distribute(
                processing_mode="distributed_epoch", service=spec.service_addr
            )
        )

    return ds


def get_dataset(spec: TFEvalSpec):
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    train_filepath = pathlib.Path(data_dir) / pathlib.Path(
        "wikitext-103/wiki.train.tokens"
    )

    return build_dataset(
        str(train_filepath),
        spec,
    )


if __name__ == "__main__":
    tf_dataset = get_dataset(TFEvalSpec(1, 1))

    for i, x in enumerate(tf_dataset):
        print(x)
        # print(x.shape)
        print(i)
        if i == 10:
            break
