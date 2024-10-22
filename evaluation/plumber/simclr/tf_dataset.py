from absl import app
from absl import flags
import time

import tensorflow as tf
import tensorflow_addons as tfa
import pathlib

import dataset_flags
from plumber_analysis import gen_util

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'time_limit_s', default=None,
    help=('Number of seconds to run for'))

flags.DEFINE_bool(
    'profile', default=False,
    help=('Whether to profile'))

DATASET_LOC = "/home/myzhao/cedar/evaluation/datasets/imagenette2/imagenette2/train/"
IMG_HEIGHT = 244
IMG_WIDTH = 244
GAUSSIAN_BLUR_KERNEL_SIZE = 11

def process_path(img):
    boxes = tf.random.uniform(shape=(1, 4))

    # img = tf.io.read_file(file_path)
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


def build_dataset(data_dir):
    ds = tf.data.Dataset.list_files(data_dir + "*/*", shuffle=True)

    # ds = tf.data.Dataset.list_files(str(data_dir / "*/*"), shuffle=True)
    ds = ds.map(tf.io.read_file, num_parallel_calls=1)
    ds = ds.map(
        lambda x: process_path(x),
        num_parallel_calls=1
    )
    ds = ds.batch(1)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def apply_options(dataset):
    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = FLAGS.dataset_threadpool_size
    options.experimental_optimization.map_and_batch_fusion = \
        FLAGS.map_and_batch_fusion
    gen_util.add_analysis_to_dataset_options(options)
    dataset = dataset.with_options(options)
    return dataset


def get_dataset():

    return build_dataset(
        DATASET_LOC,
    )

def main(_):
    tf_dataset = get_dataset()

    if FLAGS.profile:
        tf_dataset = tf_dataset.take(FLAGS.benchmark_num_elements)
        tf_dataset = apply_options(tf_dataset)
        summary = gen_util.benchmark_and_profile_dataset(
            tf_dataset, time_limit_s=FLAGS.time_limit_s
        )
    else:
        i = 0
        start_time = time.time()
        for _ in tf_dataset:
            if i % 100 == 0:
                print(i)
            i += 1
        print("Total dataset size: {}".format(i))
        end_time = time.time()
        print(end_time - start_time)

if __name__ == '__main__':
    app.run(main)