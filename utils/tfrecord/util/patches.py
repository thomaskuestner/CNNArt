import numpy as np
import random
import tensorflow as tf

def get_patches(image, num_patches=30, image_shape = [236, 320, 260],
                patch_size=[64, 64, 64], overlap = 32, start = [0, 0, 0]):

# Order: to define where to start to choose the index, from the start side or from the end side
    order = random.choice([False, True])
    index_lists = compute_patch_indices(image_shape = image_shape,
                                        patch_size = patch_size,
                                        overlap = overlap,
                                        start = start,
                                        order = order)
    assert num_patches == len(index_lists)
    patches_collection = []
    for index in index_lists:
        patch = image[index[0]:(index[0] + patch_size[0]),
                index[1]: (index[1]+ patch_size[1]), index[2]:(index[2]+ patch_size[2])]
        patches_collection.append(patch)

    patches_collection = tf.stack(patches_collection)
    #assert patches.get_shape().dims == [num_patches, patch_size, patch_size, patch_size, 1]
    return patches_collection

def compute_patch_indices(image_shape, patch_size, overlap, start = [0, 0, 0], order = True):
    if isinstance(overlap, int):
        overlap = np.asarray([overlap] * len(image_shape))
        #print(overlap)

    stop = [(i-j) for i, j  in zip(image_shape, patch_size)]
    step = patch_size - overlap
    index_list = get_set_of_patch_indices(start, stop, step, order)
    return get_random_indexs (image_shape, patch_size, index_list)

# order is for fetching those near the bounds
# if fetch in True mode, then those near the stop won't be fetched
# if fetch in False mode , then those near the start won't be fetched

def get_set_of_patch_indices(start, stop, step, order = False):
    if order:
        return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                          start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)
    else:
        return np.asarray(np.mgrid[stop[0]:start[0]:-step[0], stop[1]:start[1]:-step[1],
                           stop[2]:start[2]:-step[2]].reshape(3, -1).T, dtype=np.int)

def get_random_indexs (image_shape, patch_size, index_list):

    index0bound = image_shape[0] - patch_size[0]
    index1bound = image_shape[1] - patch_size[1]
    index2bound = image_shape[2] - patch_size[2]

    for index in index_list:
        newIndex0 = index[0] + random.randint(-10, 10)
        newIndex1 = index[1] + random.randint(-10, 10)
        newIndex2 = index[2] + random.randint(-10, 10)

        index[0] = newIndex0 if (newIndex0 <= index0bound and newIndex0 >= 0) else index[0]
        index[1] = newIndex1 if (newIndex1 <= index1bound and newIndex1 >= 0) else index[1]
        index[2] = newIndex2 if (newIndex2 <= index2bound and newIndex2 >= 0) else index[2]

    return index_list


if __name__ == '__main__':
    image_shape = [236, 320, 250]
    patch_size = [64, 64, 64]
    overlap = 32
    for _ in range(10):
        print(random.choice([True, False]))
    print(len(compute_patch_indices(image_shape, patch_size, overlap)))
    print(len(compute_patch_indices(image_shape, patch_size, overlap, order = False)))
    print(compute_patch_indices(image_shape, patch_size, overlap))