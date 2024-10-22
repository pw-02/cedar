import tensorflow as tf
import tensorflow_text as text
import pathlib

from evaluation.tf_utils import TFEvalSpec

DATASET_LOC = "datasets/wikitext103"

# from https://github.com/cirquit/presto/blob/master/openwebtext_pipeline_modern.py  # noqa: E501
# vocabulary size 50001, GPT2 originally used 50257
vocabulary_size = 50001
bpe_model_path = tf.keras.utils.get_file(
    "bpe_en_50k.model",
    "https://nlp.h-its.org/bpemb/en/en.wiki.bpe.vs50000.model",
)
bpe_model = open(bpe_model_path, "rb").read()

embedding_dimension = 768
bpe_tokernizer = text.SentencepieceTokenizer(
    model=bpe_model, out_type=tf.dtypes.int32
)

embedding = tf.Variable(
    tf.random.uniform([vocabulary_size, embedding_dimension], -1.0, 1.0)
)


def _truncate(x):
    dim = tf.shape(x)[0]
    slice_size = tf.minimum(dim, 254)
    x = tf.slice(x, [0], [slice_size])
    return x


def _embedding(x):
    return tf.nn.embedding_lookup(embedding, x)


def build_dataset(path, spec):
    # ds = _load_text(path)
    ds = tf.data.TextLineDataset(path)

    ds = ds.map(
        bpe_tokernizer.tokenize, num_parallel_calls=spec.num_parallel_calls
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
