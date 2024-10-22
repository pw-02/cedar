import tensorflow as tf
import tensorflow_text as text

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


def _tokenize(x):
    return bpe_tokernizer.tokenize(x)
