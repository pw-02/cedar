import tensorflow as tf
import fastflow as ff
import tensorflow_addons as tfa

from eval_app_runner import App

DATASET_LOC="/home/myzhao/cedar/evaluation/datasets/imagenette2/imagenette2/train/*/*"
IMG_HEIGHT = 244
IMG_WIDTH = 244
GAUSSIAN_BLUR_KERNEL_SIZE = 11

class SimCLRModel(ff.FastFlowModel):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        # do nothing
        return inputs

    def __deepcopy__(self):
        return SimCLRModel()
    
class SimCLRApp(App):
    def __init__(self, args, config):
        super().__init__(args, config)

    def process_path(self, img):
        boxes = tf.random.uniform(shape=(1, 4))

        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.expand_dims(img, axis=0)
        img = tf.image.crop_and_resize(img, boxes, [0], [IMG_HEIGHT, IMG_WIDTH])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
        img = tf.image.random_hue(img, max_delta=0.1)
        img = tf.image.rgb_to_grayscale(img)
        img = tfa.image.gaussian_filter2d(
            img,
            filter_shape=[GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_KERNEL_SIZE],
        )
        img = tf.image.per_image_standardization(img)
        return img

    def dummy_loss(self, y_true, y_pred):
        return tf.constant(0.0)

    def create_model(self):
        model = SimCLRModel()

        model.compile(optimizer="adam", loss=self.dummy_loss)
        return model

    def create_dataset(self, num_parallel):
        ds = tf.data.Dataset.list_files(DATASET_LOC, shuffle=True)
        ds = ds.map(tf.io.read_file, num_parallel_calls=num_parallel)
        ds = ds.map(
            lambda x: (self.process_path(x), tf.constant(0.0)),
            num_parallel_calls=num_parallel,
            name="prep_begin",
        )
        ds = ds.batch(1)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return ds

    def create_valid_dataset(self, num_parallel):
        return None

if __name__ == "__main__":
    app = SimCLRApp(None, None)
    ds = app.create_dataset(1)

    # for x in ds:
    #     print(x)
    #     break
    model = app.create_model()

    # config = ff.FastFlowConfig.from_yaml("/home/myzhao/FastFlow/examples/config.yaml")

    model.fit(ds, epochs=10)

