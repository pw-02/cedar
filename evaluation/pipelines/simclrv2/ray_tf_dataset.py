import pathlib
import glob
import ray
import time
import tensorflow as tf
import tensorflow_addons as tfa

DATASET_LOC = "datasets/imagenette2"
IMG_HEIGHT = 244
IMG_WIDTH = 244
GAUSSIAN_BLUR_KERNEL_SIZE = 11


def read_image(x):
    img = tf.io.read_file(x["item"])
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.expand_dims(img, axis=0)
    return {"image": img}


def convert_to_float(x):
    return {"image": tf.image.convert_image_dtype(x["image"], tf.float32)}


def crop_and_resize(x):
    boxes = tf.random.uniform(shape=(1, 4))
    return {
        "image": tf.image.crop_and_resize(
            x["image"], boxes, [0], [IMG_HEIGHT, IMG_WIDTH]
        )
    }


def random_flip(x):
    return {"image": tf.image.random_flip_left_right(x["image"])}


def color_jitter(x):
    img = tf.image.random_brightness(x["image"], max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    if img.shape[-1] == 3:
        img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
    if img.shape[-1] == 3:
        img = tf.image.random_hue(img, max_delta=0.1)
    return {"image": img}


def grayscale(x):
    return {"image": tf.image.rgb_to_grayscale(x["image"])}


def normalize(x):
    return {"image": tf.image.per_image_standardization(x["image"])}


def gaussian_blur(x):
    return {
        "image": tfa.image.gaussian_filter2d(
            x["image"],
            filter_shape=[
                GAUSSIAN_BLUR_KERNEL_SIZE,
                GAUSSIAN_BLUR_KERNEL_SIZE,
            ],
        )
    }


class Timer:
    def __init__(self):
        self._start = None
        self._end = None

    def __enter__(self):
        self._start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.perf_counter()

    def reset(self):
        self._start = time.perf_counter()

    def delta(self):
        if self._start is None or self._end is None:
            raise RuntimeError()
        return self._end - self._start


def ret_img(x):
    return x


def build_ds(root):
    # Get list of dirs
    dir_list = []
    file_list = []
    for item in pathlib.Path(root).iterdir():
        if item.is_dir():
            dir_list.append(str(item))
            files = glob.glob(f"{str(item)}/*.JPEG")
            file_list.extend(files)
    ds = ray.data.from_items(file_list)
    ds = ds.map(read_image)
    ds = ds.map(convert_to_float)
    ds = ds.map(crop_and_resize)
    ds = ds.map(random_flip)
    ds = ds.map(color_jitter)
    ds = ds.map(grayscale)
    ds = ds.map(gaussian_blur)
    ds = ds.map(normalize)
    # ds = ds.map(transform_img)

    return ds


def get_dataset():
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    train_filepath = pathlib.Path(data_dir) / pathlib.Path("imagenette2/train")

    ds = build_ds(str(train_filepath))
    ray.data.DataContext.get_current().execution_options.locality_with_output = (
        True
    )

    return ds


if __name__ == "__main__":
    ds = get_dataset()

    # for row in ds.iter_rows():
    #     print(row)
    #     break
    epoch_times = []
    for _ in range(3):
        timer = Timer()
        with timer:
            for idx, row in enumerate(ds.iter_rows()):
                pass
        epoch_times.append(timer.delta())
        print(epoch_times[-1])

    # print("Epoch times: {}".format(epoch_times))
