import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from typing import Union

AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocess(path: str):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False, dtype=tf.float32)
    img = tf.image.resize(img, [256, 256])
    img = img / 255.
    return img


def load_dataset(data_dir: Union[Path, str], batch_size):
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    file_path_list = list(sorted(data_dir.glob('*')))[::30]
    file_path_list = [str(p) for p in file_path_list]
    ds = tf.data.Dataset.from_tensor_slices(file_path_list)
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(buffer_size=1024)
    ds = ds.cache()
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds.as_numpy_iterator()


def load_mnist(batch_size):
    def preprocess(data):
        img = data['image']
        img = tf.cast(img, tf.float32)
        img = img / 255.
        return img
    builder = tfds.builder('mnist')
    builder.download_and_prepare()
    ds = builder.as_dataset(split='train', shuffle_files=True)
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(buffer_size=1024)
    ds = ds.cache()
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds.as_numpy_iterator()

