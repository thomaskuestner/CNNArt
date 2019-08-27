#!/usr/bin/env python
# coding: utf-8
#  Author: so2liu@gmail.com in Jupyter Lab

# the original version is CNNArt/utils/read_results-train-set.py
# This file should be directly called by a gadget in docker

# Input: a 2D numpy matrix
# Output: a RGB numpy matrix
import numpy as np
import os
import pydicom

def dicom_to_numpy(folderpath):
    ##  recorgnize and read IMA files from folderpath
    file_list = [f for f in os.listdir(folderpath) if os.path.isfile(os.path.join(folderpath, f)) and 'IMA' in f]
    file_list.sort()
    for each in file_list:
        print(each)
    first_img = pydicom.read_file(os.path.join(folderpath, file_list[0]))
    img_data = np.zeros((len(file_list),) + first_img.pixel_array.shape)
    for index, each_img in enumerate(file_list):
        img_data[index, :, :] = pydicom.read_file(os.path.join(folderpath, each_img)).pixel_array
    print('img_data.shape =', img_data.shape)
    return img_data

# ## Patching the image
# a pad-calculation funciton, in z,y, x constellation
def pad_size_calculate(img_shape, not_overlap_shape, overlap_shape):
    print('img_shape, not_overlap_shape, overlap_shape = ', img_shape, not_overlap_shape, overlap_shape)
    z, y, x = tuple(img_shape)
    nonover_z, nonover_y, nonover_x = tuple(not_overlap_shape)
    img_shape, not_overlap_shape, overlap_shape = np.array(img_shape), np.array(not_overlap_shape), np.array(overlap_shape)
        
    number_patches = np.ceil(np.divide(img_shape, not_overlap_shape)).astype(np.int32)
    padded_img_shape = (np.multiply(number_patches, not_overlap_shape)+overlap_shape).astype(np.int32)
    half_pad_shape = (padded_img_shape-img_shape).astype(np.float32)/2
    x_pad = (np.ceil(half_pad_shape[2]).astype(np.int32), np.floor(half_pad_shape[2]).astype(np.int32))
    y_pad = (np.ceil(half_pad_shape[1]).astype(np.int32), np.floor(half_pad_shape[1]).astype(np.int32))
    z_pad = (0, (half_pad_shape[0]*2).astype(np.int32))
    return z_pad, y_pad, x_pad, padded_img_shape


# use z y x constellation
def f3DPatching(img_origial, patch_size, overlap_rate):
    patch_size = np.flip(np.array(patch_size, dtype=np.int32), 0)  # as z y x constellation
    try:
        assert len(img_origial.shape) == 3 and len(patch_size) == 3 and overlap_rate < 1
    except Exception as e:
        print('Only support 3D image as input. The input is', img_origial.shape, patch_size, overlap_rate)
        print('Reason:', e)     
    origin_size = img_origial.shape
    overlap_pixel_yx = np.floor(overlap_rate*patch_size[1:]).astype(np.int32)
    not_overlap_pixel_yx = np.ndarray.astype(patch_size[1:]-overlap_pixel_yx, np.int32)

    # padding    
    z_pad, y_pad, x_pad, _ = pad_size_calculate(origin_size, 
                                             not_overlap_shape=(patch_size[0], not_overlap_pixel_yx[0], not_overlap_pixel_yx[1]), 
                                             overlap_shape=(0, overlap_pixel_yx[0], overlap_pixel_yx[1]))
    padded_img = np.pad(img_origial, (z_pad, y_pad, x_pad), 'constant')
    print('padded_img =', padded_img.shape)
    print('not_overlap_pixel_yx =', not_overlap_pixel_yx)
    print('overlap_pixel_yx =', overlap_pixel_yx)
    print(np.mod(padded_img.shape[1:], not_overlap_pixel_yx))
    assert not np.any(np.mod((padded_img.shape[1:]-overlap_pixel_yx-not_overlap_pixel_yx), not_overlap_pixel_yx))
#    assert not np.any(np.mod(padded_img.shape[1:], not_overlap_pixel_yx)-overlap_pixel_yx)  # ensure all zeros
    assert not np.any(np.mod(padded_img.shape[0], patch_size[0]))  # ensure all zeros    
        
    # patching 
    def fast_3D_strides(img, patch_shape, stepsize_tuple): 
        z, y, x = img.shape
        step_z, step_y, step_x = stepsize_tuple
        sz, sy, sx = img.strides
        
        patch_frame_shape = np.divide(np.array(img.shape)-(np.array(patch_shape)-np.array(stepsize_tuple)), np.array(stepsize_tuple))
        print('patch_frame_shape =', patch_frame_shape)
        patch_frame_shape = tuple(patch_frame_shape.astype(int))  # big patch struction
        result_shape = patch_frame_shape+tuple(patch_shape)
        print('result_shape =', result_shape)
        result_strides = (sx*step_z*x*y, sx*x*step_y, sx*step_x, sx*x*y, sx*x, sx)    
        return np.lib.stride_tricks.as_strided(img, result_shape, result_strides)

  
    patched_img = fast_3D_strides(padded_img, patch_shape=patch_size, 
                                  stepsize_tuple=(patch_size[0], 
                                                  not_overlap_pixel_yx[0], 
                                                  not_overlap_pixel_yx[1]))  # axis2 is full not-overlap
    patched_img = np.reshape(patched_img, (-1, )+tuple(patch_size))

    return np.swapaxes(patched_img, 1, -1)  # return (n, x, y, z) constellation


# ## Unpatching subfunction
# which returns results as the size or original image.
def f3DUnpatching(patched_img, origin_shape, patch_size, overlap_rate, overlap_method='average'):
    # input: 4d imag with (n, x, y, z) as constellation,  1d patch_size (3 elements), a float overlap_rate, method dealing with overlap => ('average', 'cutoff')
    # output: non-padded, non-seperated, non-overlapped 3d img with size of patch_size in zyx constellation
    patched_img = patched_img.copy()  # otherwise the patched_img in main() will be changed
    assert len(patched_img.shape) == 4 and isinstance(patch_size, list)
    assert overlap_method in ('average', 'cutoff')
    
    # change everything into z, y, x  constellation
    patched_img = np.swapaxes(patched_img, 1, -1)  # switch x and z
    patch_size = np.flip(np.array(patch_size, dtype=np.int32), 0)  # as z y x constellation

    # calculate pad size
    overlap_pixel_yx = np.floor(overlap_rate*patch_size[1:]).astype(np.int32)
    not_overlap_pixel_yx = (patch_size[1:]-overlap_pixel_yx).astype(np.int32)
    stepsize_tuple = (patch_size[0], not_overlap_pixel_yx[0], not_overlap_pixel_yx[1])
    z_pad, y_pad, x_pad, padded_img_shape = pad_size_calculate(origin_shape, 
                                             not_overlap_shape=(patch_size[0], not_overlap_pixel_yx[0], not_overlap_pixel_yx[1]), 
                                             overlap_shape=(0, overlap_pixel_yx[0], overlap_pixel_yx[1]))    
    rest_over_tuple = (0, overlap_pixel_yx[0], overlap_pixel_yx[1])
    patch_frame_shape = tuple(np.divide(np.array(padded_img_shape)-np.array(rest_over_tuple), np.array(stepsize_tuple)).astype(int))  # big patch struction

    non_overlap_patch_shape = (patch_size[0], not_overlap_pixel_yx[0], not_overlap_pixel_yx[1])
    non_overlap_patched_img = np.zeros((patched_img.shape[0], )+non_overlap_patch_shape)
    
    
    if overlap_method == 'average':
        for n in range(patched_img.shape[0]):
            if n%patch_frame_shape[2] == 0:
                continue
            this_img_start = patched_img[n, :, :, :overlap_pixel_yx[1]]
            last_img_end = patched_img[n-1, :, :, -overlap_pixel_yx[1]:]
#             print(n, this_img_start.shape, last_img_end.shape)
            patched_img[n, :, :, :overlap_pixel_yx[1]] = (this_img_start+last_img_end)/2
        for n in range(patched_img.shape[0]):
            if n%patch_frame_shape[1] == 0:
                continue
            this_img_start = patched_img[n, :, :overlap_pixel_yx[0], :]
            last_img_end = patched_img[n-patch_frame_shape[2], :, -overlap_pixel_yx[0]:, :]
#             print(n, this_img_start.shape, last_img_end.shape)
            patched_img[n, :, :overlap_pixel_yx[0], :] = (this_img_start+last_img_end)/2
        
        
    # padded, seperated, overlapped img => padded, seperated, non-overlapped img
    for n in range(patched_img.shape[0]):
        non_overlap_patched_img[n, :, :, :] = patched_img[n, 0:patch_size[0], 0: not_overlap_pixel_yx[0], 0: not_overlap_pixel_yx[1]]
    print('non_overlap_patched_img, patch_frame_shape, non_overlap_patch_shape =', non_overlap_patched_img.shape, patch_frame_shape, non_overlap_patch_shape)
    framed_img = np.reshape(non_overlap_patched_img, patch_frame_shape+non_overlap_patch_shape)
            
    # padded, seperated, non-overlapped img => padded, non-seperated, non-overlapped img
    result = np.ones(padded_img_shape)*255
    overlap_framed_img = np.reshape(patched_img, patch_frame_shape+tuple(patch_size))
    for fz in range(patch_frame_shape[0]):
        for fy in range(patch_frame_shape[1]-1):  # -1 for ignore the right and bottom margin
            for fx in range(patch_frame_shape[2]-1):
                result[fz*non_overlap_patch_shape[0]:(fz+1)*non_overlap_patch_shape[0], 
                       fy*non_overlap_patch_shape[1]:(fy+1)*non_overlap_patch_shape[1],
                       fx*non_overlap_patch_shape[2]:(fx+1)*non_overlap_patch_shape[2]] = framed_img[fz, fy, fx, :, :, :]
                # add the margin at right and bottom of the big img
                if fy == patch_frame_shape[1]-2:
                    result[fz*non_overlap_patch_shape[0]:(fz+1)*non_overlap_patch_shape[0], 
                           (fy+1)*non_overlap_patch_shape[1]:,
                           (fx)*non_overlap_patch_shape[2]:(fx+1)*non_overlap_patch_shape[2]] = overlap_framed_img[fz, fy+1, fx, :, :, :not_overlap_pixel_yx[1]]
                    result[fz*non_overlap_patch_shape[0]:(fz+1)*non_overlap_patch_shape[0], 
                           -patch_size[1]:,
                           -patch_size[2]:] = overlap_framed_img[fz, fy+1, fx+1, :, :, :]
                    
                if fx == patch_frame_shape[2]-2:
                    result[fz*non_overlap_patch_shape[0]:(fz+1)*non_overlap_patch_shape[0], 
                           (fy)*non_overlap_patch_shape[1]:(fy+1)*non_overlap_patch_shape[1],
                           (fx+1)*non_overlap_patch_shape[2]:] = overlap_framed_img[fz, fy, fx+1, :, :not_overlap_pixel_yx[0], :]
    
    # padded, non-seperated, non-overlapped img => non-padded, non-seperated, non-overlapped img
    result = result[0:origin_shape[0], y_pad[0]:origin_shape[1]+y_pad[0], x_pad[0]:origin_shape[2]+x_pad[0]]
    print('Unpatched.shape =', result.shape)
    assert result.shape == tuple(origin_shape)    
    return result/result.max() # normalization


# ## Make prediction using model.
def predict_with_model(patched_img, model_json, para_h5):
    # imports
    import sys
    import numpy as np                  # for algebraic operations, matrices
    import os.path                      # operating system
    import keras
    import time
    from keras import backend as K
    import tensorflow as tf

    # for avoiding CUDNN_STATUS_INTERNAL_ERROR
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.tensorflow_backend._get_available_gpus()

    def dice_coef(y_true, y_pred, epsilon=1e-5):
        dice_numerator = 2.0 * K.sum(y_true*y_pred, axis=[1,2,3,4])
        dice_denominator = K.sum(K.square(y_true), axis=[1,2,3,4]) + K.sum(K.square(y_pred), axis=[1,2,3,4])
        dice_score = dice_numerator / (dice_denominator + epsilon)
        return K.mean(dice_score, axis=0)

    def dice_coef_loss(y_true, y_pred):
        return 1-dice_coef(y_true, y_pred)

    if os.getcwd()+'/CNNArt' not in sys.path:
        sys.path.append(os.getcwd()+'/CNNArt')


#    with open(model_json, 'r+', errors='ignore') as fp:
#        model = keras.models.model_from_json(fp.read())
    print(model_json)
    json_file = open(model_json, 'r')
    loadded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loadded_model_json)

    model.load_weights(para_h5)

    prob_pre = model.predict(np.expand_dims(patched_img, axis=-1), batch_size=2, verbose=1)

    return prob_pre


def cnnart_gadgetron_interface(origin_img,
                               patch_size=[128, 128, 16],
                               overlap_rate=0.4,
                               para_h5='/misc/home/d1290/no_backup/d1290/model/16/FCN 3D-VResFCN-Upsampling final Motion Binary_3D_128x128x16_2018-10-27_19-27_weights.h5',
                               model_json='/misc/home/d1290/no_backup/d1290/model/16/FCN 3D-VResFCN-Upsampling final Motion Binary_3D_128x128x16_2018-10-27_19-27.json'):
    # test part for patching function
    origin_img = origin_img/origin_img.max()
    print('patch_size, overlap_rate = ', patch_size, overlap_rate)
    patched_img = f3DPatching(origin_img, patch_size, overlap_rate)
    print('patched_img.shape =', patched_img.shape)

    predictions = predict_with_model(patched_img, para_h5, model_json)

    predict_result = f3DUnpatching(predictions[:, :, :, :, 0], origin_img.shape, [128, 128, 16], 0.4)
    print('predict_result.shape =', predict_result.shape)
    assert predict_result.shape == origin_img.shape
    rgb_result = np.stack([origin_img, origin_img+predict_result, origin_img+(1-predict_result)], 1)  # the constellation is (n, rgb, y, x)
    print(rgb_result.shape)
    assert rgb_result.shape == origin_img.shape[:2]+(3,)+origin_img.shape[-1]
    return rgb_result


# test code
if __name__ == '__main__':
    folderpath = '/home/d1290/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol/05_fg/dicom_sorted/t1_tse_tra_Kopf_0002/'
    origin_img = dicom_to_numpy(folderpath)
    result = cnnart_gadgetron_interface(origin_img)
    print(result.shape)
    np.save('result.npy', result)
    
    pass



