import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import librosa
import random
import numpy as np

from evaluation.tf_utils import TFEvalSpec

DATASET_LOC = "datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08/en/clips/"
SAMPLE_FREQ = 8000
N_FFT = 400
FREQ_MASK_PARAM = 80
TIME_MASK_PARAM = 80
N_MELS = 256


def time_mask(x):
    t = np.random.uniform(low=0.0, high=TIME_MASK_PARAM)
    t = int(t)
    tau = x.shape[1]
    t0 = random.randint(0, tau - t)
    x[:, t0 : t0 + t] = 0
    return x


def frequency_mask(x):
    f = np.random.uniform(low=0.0, high=FREQ_MASK_PARAM)
    f = int(f)
    v = x.shape[0]
    f0 = random.randint(0, v - f)
    x[f0 : f0 + f, :] = 0
    return x


def mel(x):
    return librosa.feature.melspectrogram(
        S=x, sr=SAMPLE_FREQ, n_mels=N_MELS, n_fft=N_FFT
    )


@tf.py_function(Tout=tf.float32)
def process_path(path):
    x, sr = librosa.load(path.numpy())
    x = librosa.resample(y=x, orig_sr=sr, target_sr=SAMPLE_FREQ)
    x = np.abs(librosa.stft(x, n_fft=N_FFT)) ** 2
    x = librosa.effects.time_stretch(x, rate=0.8, n_fft=N_FFT)
    x = time_mask(x)
    x = frequency_mask(x)
    x = mel(x)
    return x


def build_dataset(data_dir, spec):
    ds = tf.data.Dataset.list_files(f"{data_dir}/*", shuffle=False)
    ds = ds.map(
        lambda x: process_path(x), num_parallel_calls=spec.num_parallel_calls
    )
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    if spec.service_addr:
        print(
            "Using tf.data.service with address {}".format(spec.service_addr)
        )
        ds = ds.apply(
            tf.data.experimental.service.distribute(
                processing_mode="distributed_epoch", service=spec.service_addr
            )
        )
    return ds


def get_dataset(spec: TFEvalSpec):
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )

    # return gen_files(train_filepath)

    return build_dataset(
        str(data_dir),
        spec,
    )


if __name__ == "__main__":
    batch_size = 8
    num_workers = tf.data.AUTOTUNE

    tf_dataset = get_dataset(TFEvalSpec(1, 1))

    for i, x in enumerate(tf_dataset):
        print(x)
        print(x.shape)

        fig, ax = plt.subplots()
        D = librosa.power_to_db(x.numpy(), ref=np.max)
        img = librosa.display.specshow(
            D, y_axis="mel", x_axis="time", sr=SAMPLE_FREQ, ax=ax
        )
        fig.savefig("tmptf.png")
        break
