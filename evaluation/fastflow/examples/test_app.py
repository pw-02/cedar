import fastflow as ff
import tensorflow as tf

from eval_app_runner import App

class TestModel(ff.FastFlowModel):
    def __init__(self):
        super().__init__()
    
    def call(self, inputs):
        # do nothing
        return inputs

    def __deepcopy__(self):
        return TestModel()

class TestApp(App):
    def __init__(self, args, config):
        super().__init__(args, config)

        self.ds = tf.data.Dataset.from_tensor_slices((tf.random.uniform([100000, 32], maxval=100, dtype=tf.int32),))


    def dummy_loss(self, y_true, y_pred):
        return tf.constant(0.0)

    def create_model(self):
        model = TestModel()

        model.compile(optimizer="adam", loss=self.dummy_loss)
        return model

    def create_dataset(self, num_parallel):
        dataset = self.ds.map(lambda x: (x+1, x), num_parallel_calls=num_parallel, name="prep_begin")
        dataset = dataset.batch(32)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    def create_valid_dataset(self, num_parallel):
        return None
        
    
def main():
    ds = dataloader()
    valid_ds = dataloader()

    model = TestModel()
    model.compile(optimizer="adam", loss=dummy_loss)

    config = ff.FastFlowConfig.from_yaml("/home/myzhao/FastFlow/eval/test/config.yaml")

    model.fit(x=ds, auto_offload_conf=config, epochs=10)


if __name__ == "__main__":
    main()