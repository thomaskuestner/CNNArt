import tensorflow as tf
from utils import Patching
from utils import tfrecord

from tensorflow.keras import backend as K


# [320, 260, 316] is the shape for NAKO dataset
# [236, 320, 260] is the shape for NAKO_IQA dataset

def tfdata_generator(file_lists, label_lists, is_training, num_parallel_calls=4, patch_size=[64, 64, 64],
                     overlap=32, image_shape=[316, 260, 320], start=[20, 60, 50],
                     num_imgaes_loaded=30, batch_size=32, num_classes = 3,
                     mean_value = 78.242,
                     std_value = 144.83,
                     prefetched_buffer_size=8000):

    # begin construct tensorflow input pipeline
    # define the patch you want to crop
    len_file_lists = len(file_lists)
    patch_size = patch_size  # size of the patches per axis
    overlap = overlap
    image_shape = image_shape
    start = start

    # number of patches to extract from each image
    num_patches = len(Patching.compute_patch_indices(image_shape=image_shape,
                                                    patch_size=patch_size,
                                                    overlap=overlap,
                                                    start=start))

    buffer_size = num_imgaes_loaded * num_patches  # shuffle patches from 34 different big images

    # This function splits a whole image into many splits and return
    get_patches_fn = lambda image: Patching.get_patches(image,
                                                       num_patches=num_patches,
                                                       image_shape=image_shape,
                                                       patch_size=patch_size,
                                                       overlap=overlap,
                                                       start=start)

    filenames = tf.data.TFRecordDataset(file_lists)
    labels = tf.data.Dataset.from_tensor_slices(label_lists)
    dataset = tf.data.Dataset.zip((filenames, labels))
    dataset = dataset.shuffle(buffer_size=len_file_lists)

    # Note: this could be rewritten as one map call
    # map parse function to each zipped element
    dataset = dataset.map(
        map_func=lambda a, b: (tfrecord.medio.convert_tf.parse_function(a), b),
        num_parallel_calls=num_parallel_calls)

    dataset = dataset.map(
        map_func=lambda a, b: (tf.cast(a, tf.float32), b),
        num_parallel_calls=num_parallel_calls)

    # normalization
    mean = tf.constant(mean_value, dtype=tf.float32)
    std = tf.constant(std_value, dtype=tf.float32)

    dataset = dataset.map(
        map_func=lambda a, b: (tf.subtract(a, mean), b),
        num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(
        map_func=lambda a, b: (tf.divide(a, std), b),
        num_parallel_calls=num_parallel_calls)

    dataset = dataset.map(map_func=lambda a, b: (get_patches_fn(a),
                                                 [b for _ in range(num_patches)]),
                          num_parallel_calls=num_parallel_calls)

    dataset = dataset.apply(tf.contrib.data.unbatch())

    dataset = dataset.map(
        map_func=lambda a, b: (tf.reshape(a, patch_size), b),
        num_parallel_calls=num_parallel_calls)

    dataset = dataset.map(
        map_func=lambda a, b: (tf.expand_dims(a, -1),
                               tf.one_hot(b, num_classes),),
        num_parallel_calls=num_parallel_calls)

    # here one could use shuffle, repeat, prefetch, ...
    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset_batched = dataset.batch(batch_size=batch_size)
    # dataset_batched = dataset
    # ???
    # dataset_batched = dataset_batched.prefetch(buffer_size = prefetched_buffer_size)
    dataset_batched = dataset_batched.repeat()

    return dataset_batched
