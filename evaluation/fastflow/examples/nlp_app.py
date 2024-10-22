import tensorflow as tf
import fastflow as ff
import tensorflow_text as text

from eval_app_runner import App

DATASET_LOC="/home/myzhao/cedar/evaluation/datasets/wikitext103/wikitext-103/wiki.train.tokens"

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
    return (tf.nn.embedding_lookup(embedding, x), tf.constant(0.0))

class WikiTextModel(ff.FastFlowModel):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        # do nothing
        return inputs

    def __deepcopy__(self):
        return WikiTextModel()
    
class WikiTextApp(App):
    def __init__(self, args, config):
        super().__init__(args, config)

    def dummy_loss(self, y_true, y_pred):
        return tf.constant(0.0)

    def create_model(self):
        model = WikiTextModel()

        model.compile(optimizer="adam", loss=self.dummy_loss)
        return model

    def create_dataset(self, num_parallel):
        ds = tf.data.TextLineDataset(DATASET_LOC).take(200000)
        ds = ds.map(
            bpe_tokernizer.tokenize, num_parallel_calls=num_parallel,
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

    # for x in ds:
    #     print(x)
    #     break
    model = app.create_model()

    # config = ff.FastFlowConfig.from_yaml("/home/myzhao/FastFlow/examples/config.yaml")

    model.fit(ds, epochs=10)

