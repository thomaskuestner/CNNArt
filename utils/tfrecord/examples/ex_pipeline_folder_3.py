#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import time
import os
import nibabel

from utils.tfrecord.medio import convert_tf
from utils.tfrecord.medio import parse_tf
from utils.tfrecord.util import patches

def parse_label(path_name):
    if path_name.find('_deep_') != -1:
        return 2
    elif path_name.find('bh') != -1:
        return 0
    elif path_name.find('fb') != -1:
        return 1


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    num_parallel_calls = 4

    # define some example values
    data_dir = '/home/s1304/no_backup/s1304/data'
    b_viewer = True
    b_verbose = True

    # choose & fetch all required data / discard subjects missing crucial data
    list_images = parse_tf.fetch_paths(data_dir, '_F_')
    print(list_images)
    list_labels = [parse_label(x) for x in list_images]


    num_samples = len(list_images)
    print('num_samples ',num_samples)

    # define the patch you want to crop
    patch_size = [128, 128, 128]         # size of the patches per axis
    overlap = 32
    image_shape = [236, 320 , 260]
    start  = [0 , 0, 50]
    # number of patches to extract from each image
    num_patches = len(patches.compute_patch_indices(image_shape = image_shape,
                                                    patch_size = patch_size,
                                                    overlap = overlap,
                                                    start = start))
    print('number of patches cropped per image: ', num_patches)

    buffer_size = 34*num_patches       # shuffle patches from 34 different big images
    batch_size = 64

    # This function splits a whole image into many splits and return
    get_patches_fn = lambda image: patches.get_patches(image,
                                                       num_patches=num_patches,
                                                       image_shape = image_shape,
                                                       patch_size=patch_size,
                                                       overlap = overlap,
                                                       start = start)

    # begin input pipeline constructs
    # generate placeholders that receive paths of type str
    images = tf.placeholder(tf.string, shape=[None])
    labels= tf.placeholder(tf.int32, shape=[None])

    # create dataset for your needs
    dataset_image = tf.data.TFRecordDataset(images)
    dataset_labels = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((dataset_image, dataset_labels))

    dataset = dataset.shuffle(buffer_size=num_samples)

    # Note: this could be rewritten as one map call
    # map parse function to each zipped element
    dataset = dataset.map(
        map_func=lambda a, b: (convert_tf.parse_function(a), b),
        num_parallel_calls=num_parallel_calls)

    # This is for single channel, extand the last axis as the channel axis
    dataset = dataset.map(
        map_func=lambda a, b: (tf.expand_dims(a, -1),
                               b,),
        num_parallel_calls=num_parallel_calls)

    dataset = dataset.map(map_func= lambda a,b : (get_patches_fn(a),
                                                  [b for _ in range(num_patches)]),
                          num_parallel_calls= num_parallel_calls)
    dataset = dataset.apply(tf.contrib.data.unbatch())

    # here one could use shuffle, repeat, prefetch, ...
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset_batched = dataset.batch(batch_size=batch_size)
    dataset_batched = dataset_batched.prefetch(buffer_size=8096)
    dataset_batched = dataset_batched.repeat()

    iterator = dataset_batched.make_initializable_iterator()
    next_element = iterator.get_next()

    # dummy "model"
    result = next_element

    # simple session example
    with tf.Session() as sess:

        sess.run(iterator.initializer,
                 feed_dict={images: list_images, labels: list_labels})
        #count = 0

        while True:
            try:
                t0 = time.time()
                #print('....')
                batched_images, train_labels = sess.run(result)

                print('elapsed loading time: ', time.time() - t0)
                #print(batched_images[0, ...].max(), batched_labels_1[0, ...].max(), batched_labels_2[0, ...].max())
                #print('count ',count)

                print(batched_images.shape)
                print(train_labels)
                '''
                # This is for viewing the patch
                image = np.array(batched_images[0])
                print(image.shape)
                nibabel.viewers.OrthoSlicer3D(image[:, :, : ,0]).show()
                '''
                #count += 1
                #if (batched_images.shape[0] != 64):
                #    count = 0

            except tf.errors.OutOfRangeError:
                print('finished')
                break


