from typing import Union, List

import tensorflow as tf


class TFTensorDontCare:
    pass


class TFOutputHint:
    def __init__(
        self,
        shape: Union[List, TFTensorDontCare],
        dtype: Union[tf.dtypes.DType, TFTensorDontCare],
    ):
        self.shape = shape
        self.dtype = dtype
