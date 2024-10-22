import pathlib
import tensorflow as tf
from tensorflow.python.ops import math_ops
import ray
import json
import time
import numpy as np

DATASET_LOC = "datasets/coco"


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
def distorted_bounding_box_crop(x):
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
    x["boxes"] = np.float32(x["boxes"])
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].
    bbox_begin, bbox_size, distort_bbox = (
        tf.image.sample_distorted_bounding_box(
            tf.shape(x["image"]),
            bounding_boxes=tf.expand_dims(x["boxes"], 0),
            min_object_covered=0.3,
            aspect_ratio_range=(0.9, 1.1),
            area_range=(0.1, 1.0),
            max_attempts=200,
            use_image_if_no_bounding_boxes=True,
        )
    )
    distort_bbox = distort_bbox[0, 0]

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(x["image"], bbox_begin, bbox_size)
    # Restore the shape since the dynamic slice loses 3rd dimension.
    cropped_image.set_shape([None, None, 3])

    # Update bounding boxes: resize and filter out.
    bboxes = bboxes_resize(distort_bbox, x["boxes"])
    labels, bboxes = bboxes_filter_overlap(x["labels"], x["boxes"])

    x["image"] = cropped_image
    x["labels"] = labels
    x["boxes"] = bboxes

    return x


def read_image(x):
    img = tf.io.read_file(x["image"])
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    x["image"] = img
    return x


def random_flip(x):
    x["image"] = tf.image.random_flip_left_right(x["image"])
    return x


def resize_image(x):
    x["image"] = tf.image.resize(
        x["image"], (1333, 1333)
    )  # Max size from Faster R-CNN
    return x


def distort(x):
    x["image"] = tf.image.random_brightness(x["image"], max_delta=32.0 / 255.0)
    x["image"] = tf.image.random_saturation(x["image"], lower=0.5, upper=1.5)
    x["image"] = tf.image.random_hue(x["image"], max_delta=0.2)
    x["image"] = tf.image.random_contrast(x["image"], lower=0.5, upper=1.5)

    return x


def normalize(x):
    x["image"] = x["image"] * 255.0

    mean = tf.constant([123.0, 117.0, 104.0], dtype=x["image"].dtype)
    x["image"] = x["image"] - mean
    return x


def build_ds(data_dir):
    ann_file = data_dir + "/annotations/instances_val2017.json"

    annotations = {}
    imgs = []

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

        sample = {
            "image": data_dir + "/val2017/" + img["file_name"],
            "labels": labels,
            "boxes": boxes,
        }

        imgs.append(sample)

    ds = ray.data.from_items(imgs)
    ds = ds.map(read_image)
    ds = ds.map(distorted_bounding_box_crop)
    ds = ds.map(resize_image)
    ds = ds.map(random_flip)
    ds = ds.map(distort)
    ds = ds.map(normalize)

    return ds


def get_dataset():
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    ds = build_ds(str(data_dir))
    ray.data.DataContext.get_current().execution_options.locality_with_output = (
        True
    )

    return ds


if __name__ == "__main__":
    ds = get_dataset()
    epoch_times = []

    for _ in range(3):
        timer = Timer()
        with timer:
            for idx, row in enumerate(ds.iter_rows()):
                pass
        epoch_times.append(timer.delta())
        print(epoch_times[-1])

    print("Epoch times: {}".format(epoch_times))
