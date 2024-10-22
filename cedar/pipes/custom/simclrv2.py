import tensorflow as tf
import tensorflow_addons as tfa

IMG_HEIGHT = 244
IMG_WIDTH = 244
GAUSSIAN_BLUR_KERNEL_SIZE = 11


def read_image(x):
    img = tf.io.read_file(x)
    return img


def decode_jpeg(x):
    img = tf.image.decode_jpeg(x, channels=3)
    img = tf.expand_dims(img, axis=0)
    return img


def convert_to_float(x):
    return tf.image.convert_image_dtype(x, tf.float32)


def crop_and_resize(x):
    boxes = tf.random.uniform(shape=(1, 4))
    return tf.image.crop_and_resize(x, boxes, [0], [IMG_HEIGHT, IMG_WIDTH])


def random_flip(x):
    return tf.image.random_flip_left_right(x)


def color_jitter(x):
    img = tf.image.random_brightness(x, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    if img.shape[-1] == 3:
        img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
    if img.shape[-1] == 3:
        img = tf.image.random_hue(img, max_delta=0.1)
    return img


def gaussian_blur(x):
    return tfa.image.gaussian_filter2d(
        x,
        filter_shape=[GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_KERNEL_SIZE],
    )
