import tensorflow as tf


from utils.tfrecord.medio import convert_tf, parse_tf
from utils.tfrecord.util import patches
# It shall return the data associated label


def parse_label(path_name):
    if path_name.find('_deep_') != -1:
        return 2
    elif path_name.find('bh') != -1:
        return 0
    elif path_name.find('fb') != -1:
        return 1

def create_dataset (data_dir, num_parallel_calls = 4, patch_size=[64, 64, 64],
                    overlap=32, image_shape=[236 , 320, 260], start=[0, 0, 50],
                    num_imgaes_loaded=32, batch_size = 64, prefetched_buffer_size=8000):
    NUM_CLASSES = 3

    # choose & fetch all required data / discard subjects missing crucial data
    list_images = parse_tf.fetch_paths(data_dir, '_F_')
    print(list_images)
    list_labels = [parse_label(x) for x in list_images]
    # print(list_labels)

    num_samples = len(list_images)
    print('num_samples ', num_samples)

    # define the patch you want to crop
    patch_size = patch_size  # size of the patches per axis
    overlap = overlap
    image_shape = image_shape
    start = start
    # number of patches to extract from each image
    num_patches = len(patches.compute_patch_indices(image_shape=image_shape,
                                                    patch_size=patch_size,
                                                    overlap=overlap,
                                                    start=start))
    print('number of patches cropped per image: ', num_patches)

    buffer_size = num_imgaes_loaded * num_patches  # shuffle patches from 34 different big images

    # This function splits a whole image into many splits and return
    get_patches_fn = lambda image: patches.get_patches(image,
                                                       num_patches=num_patches,
                                                       image_shape=image_shape,
                                                       patch_size=patch_size,
                                                       overlap=overlap,
                                                       start=start)

    # generate placeholders that receive paths of type str
    images = tf.placeholder(tf.string, shape=[None])
    labels = tf.placeholder(tf.int32, shape=[None])

    # create dataset for your needs
    dataset_image = tf.data.TFRecordDataset(images)
    dataset_labels = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((dataset_image, dataset_labels))

    # Note: this could be rewritten as one map call
    # map parse function to each zipped element
    dataset = dataset.map(
        map_func=lambda a, b: (convert_tf.parse_function(a),b),
        num_parallel_calls=num_parallel_calls)

    # This is for single channel, extand the last axis as the channel axis
    dataset = dataset.map(
        map_func=lambda a, b: (tf.expand_dims(a, -1),
                               tf.one_hot(b, NUM_CLASSES),),
        num_parallel_calls=num_parallel_calls)

    dataset = dataset.map(map_func=lambda a, b: (get_patches_fn(a),
                                                 [b for _ in range(num_patches)]),
                          num_parallel_calls=num_parallel_calls)
    dataset = dataset.apply(tf.contrib.data.unbatch())

    # here one could use shuffle, repeat, prefetch, ...
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset_batched = dataset.batch(batch_size=batch_size)
    dataset_batched = dataset_batched.prefetch(buffer_size = prefetched_buffer_size)
    dataset_batched = dataset_batched.repeat()

    iterator =dataset_batched.make_initializable_iterator()

    next_element = iterator.get_next()

    # dummy "model"
    result = next_element


    with tf.Session() as sess:

        sess.run(iterator.initializer,
                 feed_dict={images: list_images, labels: list_labels})

        while True:
            try:
                yield sess.run(result)

            except tf.errors.OutOfRangeError:
                print('finished')
                break
