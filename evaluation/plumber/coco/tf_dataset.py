from absl import app
from absl import flags
import time
import json

import tensorflow as tf
import pathlib
from tensorflow.python.ops import math_ops

import dataset_flags
from plumber_analysis import gen_util

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'time_limit_s', default=None,
    help=('Number of seconds to run for'))

flags.DEFINE_bool(
    'profile', default=False,
    help=('Whether to profile'))

DATASET_LOC = "/home/myzhao/cedar/evaluation/datasets/coco"



def bboxes_resize(bbox_ref, bboxes, name=None):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform. Useful for
    updating a collection of boxes after cropping an image.
    """
    # Bboxes is dictionary.
    if isinstance(bboxes, dict):
        d_bboxes = {}
        for c in bboxes.keys():
            d_bboxes[c] = bboxes_resize(bbox_ref, bboxes[c])
        return d_bboxes

    # Tensors inputs.
    # Translate.
    v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
    bboxes = bboxes - v
    # Scale.
    s = tf.stack(
        [
            bbox_ref[2] - bbox_ref[0],
            bbox_ref[3] - bbox_ref[1],
            bbox_ref[2] - bbox_ref[0],
            bbox_ref[3] - bbox_ref[1],
        ]
    )
    bboxes = bboxes / s
    return bboxes


def safe_divide(numerator, denominator, name):
    """Divides two values, returning 0 if the denominator is <= 0.
    Args:
      numerator: A real `Tensor`.
      denominator: A real `Tensor`, with dtype matching `numerator`.
      name: Name for the returned op.
    Returns:
      0 if `denominator` <= 0, else `numerator` / `denominator`
    """
    return tf.where(
        math_ops.greater(denominator, 0),
        math_ops.divide(numerator, denominator),
        tf.zeros_like(numerator),
        name=name,
    )


def bboxes_intersection(bbox_ref, bboxes):
    """Compute relative intersection between a reference box and a
    collection of bounding boxes. Namely, compute the quotient between
    intersection area and box area.

    Args:
      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
      bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
      (N,) Tensor with relative intersection.
    """
    # Should be more efficient to first transpose.
    bboxes = tf.transpose(bboxes)
    bbox_ref = tf.transpose(bbox_ref)
    # Intersection bbox and volume.
    int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
    int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
    int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
    int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
    h = tf.maximum(int_ymax - int_ymin, 0.0)
    w = tf.maximum(int_xmax - int_xmin, 0.0)
    # Volumes.
    inter_vol = h * w
    bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
    scores = safe_divide(inter_vol, bboxes_vol, "intersection")
    return scores


def bboxes_filter_overlap(labels, bboxes):
    """Filter out bounding boxes based on (relative )overlap with reference
    box [0, 0, 1, 1].  Remove completely bounding boxes, or assign negative
    labels to the one outside (useful for latter processing...).

    Return:
      labels, bboxes: Filtered (or newly assigned) elements.
    """
    scores = bboxes_intersection(
        tf.constant([0, 0, 1, 1], bboxes.dtype), bboxes
    )
    mask = scores > 0.5
    # Specify shape of mask
    mask.set_shape([None])
    labels = tf.boolean_mask(labels, mask)
    bboxes = tf.boolean_mask(bboxes, mask)
    return labels, bboxes


# @tf.py_function(Tout=(tf.float32, tf.int32, tf.float32))
def distorted_bounding_box_crop(
    image,
    labels,
    bboxes,
):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].
    bbox_begin, bbox_size, distort_bbox = (
        tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(bboxes, 0),
            min_object_covered=0.3,
            aspect_ratio_range=(0.9, 1.1),
            area_range=(0.1, 1.0),
            max_attempts=200,
            use_image_if_no_bounding_boxes=True,
        )
    )
    distort_bbox = distort_bbox[0, 0]

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    # Restore the shape since the dynamic slice loses 3rd dimension.
    cropped_image.set_shape([None, None, 3])

    # Update bounding boxes: resize and filter out.
    bboxes = bboxes_resize(distort_bbox, bboxes)
    labels, bboxes = bboxes_filter_overlap(labels, bboxes)
    return cropped_image, labels, bboxes


def read_image(img, labels, boxes, pad_len):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Drop the padding
    labels = labels[:pad_len]
    boxes = boxes[:pad_len]

    return img, labels, boxes


def random_flip(img, labels, boxes):
    img = tf.image.random_flip_left_right(img)
    return img, labels, boxes


def resize_image(img, labels, boxes):
    img = tf.image.resize(img, (1333, 1333))  # Max size from Faster R-CNN
    return img, labels, boxes


def distort(image, labels, boxes):
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    return image, labels, boxes


def normalize(image, labels, boxes):
    image = image * 255.0
    if image.get_shape().ndims != 3:
        raise ValueError("Input must be of size [height, width, C>0]")
    num_channels = image.get_shape().as_list()[-1]
    if 3 != num_channels:
        raise ValueError("len(means) must match the number of channels")

    mean = tf.constant([123.0, 117.0, 104.0], dtype=image.dtype)
    image = image - mean
    return image, labels, boxes


def build_dataset(data_dir):
    ann_file = data_dir + "/annotations/instances_val2017.json"

    annotations = {}
    imgs = []

    max_len = 0

    with open(ann_file, "r") as f:
        ann_json = json.load(f)

    for ann in ann_json["annotations"]:
        img_id = ann["image_id"]
        if img_id in annotations:
            annotations[img_id].append(ann)
        else:
            annotations[img_id] = [ann]

    for img in ann_json["images"]:
        img_id = img["id"]
        if img_id not in annotations:
            continue
        boxes = []
        labels = []
        for ann in annotations[img_id]:
            x, y, w, h = ann["bbox"]
            # Normalize to width and height of image
            x1 = x / img["width"]
            y1 = y / img["height"]
            x2 = (x + w) / img["width"]
            y2 = (y + h) / img["height"]

            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])

        if len(boxes) > max_len:
            max_len = len(boxes)

        sample = (
            data_dir + "/val2017/" + img["file_name"],
            labels,
            boxes,
        )

        imgs.append(sample)

    # Pad labels and boxes - tf service can't support generators
    for i, img in enumerate(imgs):
        assert len(img[1]) == len(img[2])
        pad_len = len(img[1])

        new_label = img[1] + [0] * (max_len - len(img[1]))
        new_boxes = img[2] + [[0, 0, 0, 0]] * (max_len - len(img[2]))
        imgs[i] = (img[0], new_label, new_boxes, pad_len)

    # Convert imgs to tensor
    # Create tf tensor of strings
    img_paths = tf.ragged.constant([img[0] for img in imgs])
    labels = tf.constant([img[1] for img in imgs])
    boxes = tf.constant([img[2] for img in imgs])
    pad_lens = tf.constant([img[3] for img in imgs])

    ds = tf.data.Dataset.from_tensor_slices(
        (img_paths, labels, boxes, pad_lens)
    )

    ds = ds.map(read_image, num_parallel_calls=None)
    ds = ds.map(
        distorted_bounding_box_crop, num_parallel_calls=None
    )
    ds = ds.map(resize_image, num_parallel_calls=None)
    ds = ds.map(random_flip, num_parallel_calls=None)
    ds = ds.map(distort, num_parallel_calls=None)
    ds = ds.map(normalize, num_parallel_calls=None)
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
        for x in tf_dataset:
            if i % 100 == 0:
                print(x)
                break
            i += 1
        print("Total dataset size: {}".format(i))
        end_time = time.time()
        print(end_time - start_time)

if __name__ == '__main__':
    app.run(main)