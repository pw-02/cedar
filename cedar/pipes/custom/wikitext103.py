import tensorflow as tf
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

embedding = tf.Variable(tf.random.uniform([50257, 764], -1.0, 1.0))


@tf.py_function(Tout=tf.int32)
def _tokenize(x):
    return tokenizer(str(x.numpy()), return_tensors="tf")["input_ids"]


def _embedding(x):
    return tf.nn.embedding_lookup(embedding, x)


def _truncate(x):
    dim = tf.shape(x)[1]
    slice_size = tf.minimum(dim, 254)
    x = tf.slice(x, [0, 0], [1, slice_size])
    return x
