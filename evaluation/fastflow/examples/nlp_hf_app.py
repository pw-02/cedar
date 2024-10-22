import tensorflow as tf
import fastflow as ff
from transformers import GPT2Tokenizer

from eval_app_runner import App

DATASET_LOC="/home/myzhao/cedar/evaluation/datasets/wikitext103/wikitext-103/wiki.train.tokens"

def _load_text(path):
    text = tf.io.read_file(path)
    return tf.data.Dataset.from_tensor_slices(tf.strings.split(text, "\n"))


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
embedding = tf.Variable(tf.random.uniform([50257, 764], -1.0, 1.0))


def _tokenize(x):
    return tokenizer(str(x.numpy()), return_tensors="tf")["input_ids"]

def tokenize(x):
    res = tf.py_function(_tokenize, [x], [tf.int32])
    return res

def _truncate(x):
    dim = tf.shape(x)[1]
    slice_size = tf.minimum(dim, 254)
    x = tf.slice(x, [0, 0], [1, slice_size])
    return x


def _embedding(x):
    return (tf.nn.embedding_lookup(embedding, x), tf.constant(0.0))


class WikiTextModel(ff.FastFlowModel):
# class WikiTextModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        # do nothing
        return inputs

    def __deepcopy__(self):
        return WikiTextModel()
    
class WikiTextApp(App):
# class WikiTextApp():
    def __init__(self, args, config):
        super().__init__(args, config)
        # pass

    def dummy_loss(self, y_true, y_pred):
        return tf.constant(0.0)

    def create_model(self):
        model = WikiTextModel()

        model.compile(optimizer="adam", loss=self.dummy_loss)
        return model

    def create_dataset(self, num_parallel):
        ds = tf.data.TextLineDataset(DATASET_LOC).take(100000)
        ds = ds.map(
            tokenize,
            num_parallel_calls=num_parallel,
            name="prep_begin",
        )
        ds = ds.map(_truncate, num_parallel_calls=num_parallel)
        ds = ds.map(_embedding, num_parallel_calls=num_parallel)
        ds = ds.batch(1)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    def create_valid_dataset(self, num_parallel):
        return None

if __name__ == "__main__":
    app = WikiTextApp(None, None)
    ds = app.create_dataset(1)

    for x in ds:
        print(x)
        break
    model = app.create_model()

    # config = ff.FastFlowConfig.from_yaml("/home/myzhao/FastFlow/examples/config.yaml")

    model.fit(ds, epochs=10)

