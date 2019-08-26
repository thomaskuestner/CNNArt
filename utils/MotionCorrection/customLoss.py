import sys
from keras.layers import concatenate, Lambda, Input
from keras.metrics import mse
from keras import backend as K
from keras.applications.vgg19 import VGG19
from keras.models import Model, load_model
import numpy as np


def preprocessing(inputs):
    output = Lambda(lambda x: (x - K.min(x)) * 255 / (K.max(x) - K.min(x) + K.epsilon()),
                    output_shape=inputs._keras_shape)(inputs)
    # output = 255 * inputs
    K.update_sub(output[:, 0, :, :], 123.68)
    K.update_sub(output[:, 1, :, :], 116.779)
    K.update_sub(output[:, 2, :, :], 103.939)

    return output[:, ::-1, :, :]


def reshape(inputs, patchSize):
    return K.reshape(inputs, (-1, 1, patchSize[0], patchSize[1]))


def compute_mse_loss(dHyper, dParam, x_ref, decoded_ref2ref, decoded_art2ref):
    if len(dParam['patchSize']) == 2:
        loss_ref2ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,)) \
            ([Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref),
              Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(decoded_ref2ref)])
        loss_art2ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,)) \
            ([Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref),
              Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_art2ref._keras_shape)(decoded_art2ref)])

    elif len(dParam['patchSize']) == 3:
        loss_ref2ref = 0
        loss_art2ref = 0

    return loss_ref2ref, loss_art2ref


def compute_abs_loss(dHyper, dParam, x_ref, decoded_ref2ref, decoded_art2ref):
    if len(dParam['patchSize']) == 2:
        loss_ref2ref = Lambda(lambda x: K.mean(K.sum(K.abs(x[0] - x[1]), [1, 2, 3])), output_shape=(None,)) \
        ([(x_ref), (decoded_ref2ref)])

        loss_art2ref = Lambda(lambda x: K.mean(K.sum(K.abs(x[0] - x[1]), [1, 2, 3])), output_shape=(None,)) \
        ([(x_ref), (decoded_art2ref)])

    elif len(dParam['patchSize']) == 3:
        loss_ref2ref = Lambda(lambda x: K.mean(K.sum(K.abs(x[0] - x[1]), [1, 2, 3, 4])), output_shape=(None,)) \
            ([(x_ref), (decoded_ref2ref)])

        loss_art2ref = Lambda(lambda x: K.mean(K.sum(K.abs(x[0] - x[1]), [1, 2, 3, 4])), output_shape=(None,)) \
            ([(x_ref), (decoded_art2ref)])



    return loss_ref2ref, loss_art2ref


# def compute_MSSIM_loss(dHyper, dParam, x_ref, decoded_ref2ref, decoded_art2ref):
#     C1 = (1 * 0.01) ** 2
#     C2 = (1 * 0.03) ** 2
#
#     sigma = 1.5
#     num = 16  # factor of patchSize
#     kernalSize = int(dParam['patchSize'][0] / num)
#
#    # initialize the gaussian filter based on the bottom size
#     width = kernalSize - 1
#     w = np.exp(-1. * np.arange(-(width / 2), width / 2 + 1) ** 2 / (2 * sigma ** 2))
#     w = np.outer(w, w.reshape((kernalSize, 1)))  # extend to 2D
#     w = np.tile(w, (num, num))
#     w = w / np.sum(w)  # normailization
#     w = np.reshape(w, (1, 1, dParam['patchSize'][0], dParam['patchSize'][0]))  # reshape to 4D
#     w = np.tile(w, (dParam['batchSize'][0], 1, 1, 1))
#     w = K.variable(value=w)
#
#     mu_x = Lambda(lambda x: K.sum(w * x, axis=(2, 3), keepdims=True), output_shape=(None,)) \
#         (Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref))
#     sigma_x2 = Lambda(lambda x: K.sum(w * K.square(x - mu_x), axis=(2, 3)), output_shape=(None,)) \
#         (Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref))
#
#    # art2ref
#     mu_y = Lambda(lambda x: K.sum(w * x, axis=(2, 3), keepdims=True), output_shape=(None,)) \
#         (Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_art2ref._keras_shape)(decoded_art2ref))
#     sigma_y2 = Lambda(lambda x: K.sum(w * K.square(x - mu_y), axis=(2, 3)), output_shape=(None,)) \
#         (Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(decoded_art2ref))
#     sigma_xy = Lambda(lambda x: K.sum(w * (x[0] - mu_x) * (x[1] - mu_y), axis=(2, 3)), output_shape=(None,)) \
#         ([Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref),
#           Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_art2ref._keras_shape)(decoded_art2ref)])
#
#     l = (2 * mu_x * mu_y + C1) / (K.square(mu_x) + K.square(mu_y) + C1)
#     cs = (2 * K.abs(sigma_xy) + C2) / (sigma_x2 + sigma_y2 + C2)
#
#     loss_art2ref = (1 - K.mean(l * cs)) / 2
#
#    # ref2ref
#     mu_y = Lambda(lambda x: K.sum(w * x, axis=(2, 3), keepdims=True), output_shape=(None,)) \
#         (Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(decoded_ref2ref))
#     sigma_y2 = Lambda(lambda x: K.sum(w * K.square(x - mu_y), axis=(2, 3)), output_shape=(None,)) \
#         (Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(decoded_ref2ref))
#     sigma_xy = Lambda(lambda x: K.sum(w * (x[0] - mu_x) * (x[1] - mu_y), axis=(2, 3)), output_shape=(None,)) \
#         ([Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref),
#           Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(decoded_ref2ref)])
#
#     l = (2 * mu_x * mu_y + C1) / (K.square(mu_x) + K.square(mu_y) + C1)
#     cs = (2 * K.abs(sigma_xy) + C2) / (sigma_x2 + sigma_y2 + C2)
#
#     loss_ref2ref = (1 - K.mean(l * cs)) / 2
#
#     return loss_ref2ref, loss_art2ref


def compute_MS_SSIM_loss(dHyper, dParam, x_ref, decoded_ref2ref, decoded_art2ref):
    C1 = (1 * 0.01) ** 2
    C2 = (1 * 0.03) ** 2

    sigma = (0.5, 1., 2., 4., 8.)
    num_scale = len(sigma)
    num = 16
    kernalSize = int(dParam['patchSize'][0] / num)

    if len(dParam['patchSize']) == 2:
        w = np.empty((num_scale, dParam['batchSize'][0], 1, dParam['patchSize'][0], dParam['patchSize'][1]))

        for i in range(num_scale):
            width = kernalSize - 1
            weights = np.exp(-1. * np.arange(-int(width / 2), int(width / 2) + 1) ** 2 / (2 * sigma[i] ** 2))
            weights = np.outer(weights, weights.reshape((kernalSize, 1)))  # extend to 2D
            weights = np.tile(weights, (num, num))
            weights = weights / np.sum(weights)  # normailization
            weights = np.reshape(weights, (1, 1, dParam['patchSize'][0], dParam['patchSize'][0]))  # reshape to 4D
            weights = np.tile(weights, (dParam['batchSize'][0], 1, 1, 1))
            w[i, :, :, :, :] = weights

        w = K.variable(value=w)

        # tile the input to 5D
        x_ref = K.tile(x_ref, (num_scale, 1, 1, 1, 1))
        decoded_ref2ref = K.tile(decoded_ref2ref, (num_scale, 1, 1, 1, 1))
        decoded_art2ref = K.tile(decoded_art2ref, (num_scale, 1, 1, 1, 1))

        mu_x = Lambda(lambda x: K.sum(w * x, axis=(3, 4), keepdims=True), output_shape=(None,)) \
            (Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref))
        sigma_x2 = Lambda(lambda x: K.sum(w * K.square(x - mu_x), axis=(3, 4)), output_shape=(None,)) \
            (Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref))

        # art2ref
        mu_y = Lambda(lambda x: K.sum(w * x, axis=(3, 4), keepdims=True), output_shape=(None,)) \
            (Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_art2ref._keras_shape)(decoded_art2ref))
        sigma_y2 = Lambda(lambda x: K.sum(w * K.square(x - mu_y), axis=(3, 4)), output_shape=(None,)) \
            (Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(decoded_art2ref))
        sigma_xy = Lambda(lambda x: K.sum(w * (x[0] - mu_x) * (x[1] - mu_y), axis=(3, 4)), output_shape=(None,)) \
            ([Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref),
              Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_art2ref._keras_shape)(decoded_art2ref)])

        l = (2 * mu_x * mu_y + C1) / (K.square(mu_x) + K.square(mu_y) + C1)
        cs = (2 * K.abs(sigma_xy) + C2) / (sigma_x2 + sigma_y2 + C2)
        Pcs = K.prod(cs, axis=0)
        loss_art2ref = (1 - K.mean(l[-1, :, :] * Pcs)) / 2

        # ref2ref
        mu_y = Lambda(lambda x: K.sum(w * x, axis=(3, 4), keepdims=True), output_shape=(None,)) \
            (Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(decoded_ref2ref))
        sigma_y2 = Lambda(lambda x: K.sum(w * K.square(x - mu_y), axis=(3, 4)), output_shape=(None,)) \
            (Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(decoded_ref2ref))
        sigma_xy = Lambda(lambda x: K.sum(w * (x[0] - mu_x) * (x[1] - mu_y), axis=(3, 4)), output_shape=(None,)) \
            ([Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref),
              Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(decoded_ref2ref)])

        l = (2 * mu_x * mu_y + C1) / (K.square(mu_x) + K.square(mu_y) + C1)
        cs = (2 * K.abs(sigma_xy) + C2) / (sigma_x2 + sigma_y2 + C2)
        Pcs = K.prod(cs, axis=0)
        loss_ref2ref = (1 - K.mean(l[-1, :, :] * Pcs)) / 2

    elif len(dParam['patchSize']) == 3:
        depth = dParam['patchSize'][2]
        w = np.empty((num_scale, dParam['batchSize'][0], 1, 48, 48, depth))

        for i in range(num_scale):
            width = kernalSize - 1
            weights = np.exp(-1. * np.arange(-int(width / 2), int(width / 2) + 1) ** 2 / (2 * sigma[i] ** 2))
            weights = np.outer(weights, weights.reshape((kernalSize, 1)))  # extend to 2D
            weights = np.tile(weights, (num, num))
            weights = weights / np.sum(weights)  # normailization
            weights = np.reshape(weights, (1, 1, dParam['patchSize'][0], dParam['patchSize'][0], 1))  # reshape to 5D
            weights = np.tile(weights, (dParam['batchSize'][0], 1, 1, 1, depth))
            w[i, :, :, :, :, :] = weights

        w = K.variable(value=w)

        # tile the input to 6D
        x_ref = K.tile(x_ref, (num_scale, 1, 1, 1, 1, 1))
        decoded_ref2ref = K.tile(decoded_ref2ref, (num_scale, 1, 1, 1, 1, 1))
        decoded_art2ref = K.tile(decoded_art2ref, (num_scale, 1, 1, 1, 1, 1))

        mu_x = Lambda(lambda x: K.sum(w * x, axis=(3, 4, 5), keepdims=True), output_shape=(None,)) \
            (Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref))
        sigma_x2 = Lambda(lambda x: K.sum(w * K.square(x - mu_x), axis=(3, 4, 5)), output_shape=(None,)) \
            (Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref))

        # art2ref
        mu_y = Lambda(lambda x: K.sum(w * x, axis=(3, 4, 5), keepdims=True), output_shape=(None,)) \
            (Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_art2ref._keras_shape)(decoded_art2ref))
        sigma_y2 = Lambda(lambda x: K.sum(w * K.square(x - mu_y), axis=(3, 4, 5)), output_shape=(None,)) \
            (Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(decoded_art2ref))
        sigma_xy = Lambda(lambda x: K.sum(w * (x[0] - mu_x) * (x[1] - mu_y), axis=(3, 4, 5)), output_shape=(None,)) \
            ([Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref),
              Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_art2ref._keras_shape)(decoded_art2ref)])

        l = (2 * mu_x * mu_y + C1) / (K.square(mu_x) + K.square(mu_y) + C1)
        cs = (2 * K.abs(sigma_xy) + C2) / (sigma_x2 + sigma_y2 + C2)
        Pcs = K.prod(cs, axis=0)
        loss_art2ref = (1 - K.mean(l[-1, :, :] * Pcs)) / 2

        # ref2ref
        mu_y = Lambda(lambda x: K.sum(w * x, axis=(3, 4, 5), keepdims=True), output_shape=(None,)) \
            (Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(decoded_ref2ref))
        sigma_y2 = Lambda(lambda x: K.sum(w * K.square(x - mu_y), axis=(3, 4, 5)), output_shape=(None,)) \
            (Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(decoded_ref2ref))
        sigma_xy = Lambda(lambda x: K.sum(w * (x[0] - mu_x) * (x[1] - mu_y), axis=(3, 4, 5)), output_shape=(None,)) \
            ([Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref),
              Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(decoded_ref2ref)])

        l = (2 * mu_x * mu_y + C1) / (K.square(mu_x) + K.square(mu_y) + C1)
        cs = (2 * K.abs(sigma_xy) + C2) / (sigma_x2 + sigma_y2 + C2)
        Pcs = K.prod(cs, axis=0)
        loss_ref2ref = (1 - K.mean(l[-1, :, :] * Pcs)) / 2

    return loss_ref2ref, loss_art2ref


def compute_charbonnier_loss(dHyper, dParam, x_ref, decoded_ref2ref, decoded_art2ref):
    epsilon = 0.1
    if len(dParam['patchSize']) == 2:
        loss_ref2ref = Lambda(lambda x: K.mean(K.sum(K.sqrt(K.square(x[0] - x[1]) + epsilon ** 2), [1, 2, 3])),
                              output_shape=(None,)) \
            ([Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref),
              Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(decoded_ref2ref)])

        loss_art2ref = Lambda(lambda x: K.mean(K.sum(K.sqrt(K.square(x[0] - x[1]) + epsilon ** 2), [1, 2, 3])),
                              output_shape=(None,)) \
            ([Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref),
              Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_art2ref._keras_shape)(decoded_art2ref)])

    if len(dParam['patchSize']) == 3:
        loss_ref2ref = Lambda(lambda x: K.mean(K.sum(K.sqrt(K.square(x[0] - x[1]) + epsilon ** 2), [1, 2, 3, 4])),
                              output_shape=(None,)) \
            ([Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref),
              Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(decoded_ref2ref)])

        loss_art2ref = Lambda(lambda x: K.mean(K.sum(K.sqrt(K.square(x[0] - x[1]) + epsilon ** 2), [1, 2, 3, 4])),
                              output_shape=(None,)) \
            ([Lambda(lambda x: dHyper['nScale'] * x, output_shape=x_ref._keras_shape)(x_ref),
              Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_art2ref._keras_shape)(decoded_art2ref)])

    return loss_ref2ref, loss_art2ref


def compute_gradient_entropy(dHyper, dParam, decoded_ref2ref, decoded_art2ref, patchSize):
    if len(dParam['patchSize']) == 2:
        decoded_ref2ref = Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(
            decoded_ref2ref)
        a_ref2ref = K.square(
            decoded_ref2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1] - decoded_ref2ref[:, :, 1:, :patchSize[1] - 1])
        b_ref2ref = K.square(
            decoded_ref2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1] - decoded_ref2ref[:, :, :patchSize[0] - 1, 1:])
        # (128, 1, 47, 47)
        gradient_ref2ref = K.sqrt(a_ref2ref + b_ref2ref + K.epsilon())
        # (128,)
        sum_gradient_ref2ref = K.sum(gradient_ref2ref, [1, 2, 3])
        # (128, 1, 1, 1)
        sum_gradient_ref2ref = K.reshape(sum_gradient_ref2ref, shape=(gradient_ref2ref.shape[0], 1, 1, 1))
        # (128, 1, 47, 47)
        h_ref2ref = gradient_ref2ref / sum_gradient_ref2ref
        # (1,)
        ge_ref2ref = K.mean(K.sum(-h_ref2ref * K.log(K.clip(h_ref2ref, K.epsilon(), None) + 1.), [1, 2, 3]))

        decoded_art2ref = Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_art2ref._keras_shape)(
            decoded_art2ref)
        a_art2ref = K.square(
            decoded_art2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1] - decoded_art2ref[:, :, 1:, :patchSize[1] - 1])
        b_art2ref = K.square(
            decoded_art2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1] - decoded_art2ref[:, :, :patchSize[0] - 1, 1:])
        # (128, 1, 47, 47)
        gradient_art2ref = K.sqrt(a_art2ref + b_art2ref + K.epsilon())
        # (128,)
        sum_gradient_art2ref = K.sum(gradient_art2ref, [1, 2, 3])
        # (128, 1, 1, 1)
        sum_gradient_art2ref = K.reshape(sum_gradient_art2ref, shape=(gradient_art2ref.shape[0], 1, 1, 1))
        # (128, 1, 47, 47)
        h_art2ref = gradient_art2ref / sum_gradient_art2ref
        # (1,)
        ge_art2ref = K.mean(K.sum(-h_art2ref * K.log(K.clip(h_art2ref, K.epsilon(), None) + 1.), [1, 2, 3]))

    elif len(dParam['patchSize']) == 3:
        decoded_ref2ref = Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(
            decoded_ref2ref)
        a_ref2ref = K.square(
            decoded_ref2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1, :] - decoded_ref2ref[:, :, 1:, :patchSize[1] - 1, :])
        b_ref2ref = K.square(
            decoded_ref2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1, :] - decoded_ref2ref[:, :, :patchSize[0] - 1, 1:, :])
        # (128, 1, 47, 47)
        gradient_ref2ref = K.sqrt(a_ref2ref + b_ref2ref + K.epsilon())
        # (128,)
        sum_gradient_ref2ref = K.sum(gradient_ref2ref, [1, 2, 3, 4])
        # (128, 1, 1, 1)
        sum_gradient_ref2ref = K.reshape(sum_gradient_ref2ref, shape=(gradient_ref2ref.shape[0], 1, 1, 1, 1))
        # (128, 1, 47, 47)
        h_ref2ref = gradient_ref2ref / sum_gradient_ref2ref
        # (1,)
        ge_ref2ref = K.mean(K.sum(-h_ref2ref * K.log(K.clip(h_ref2ref, K.epsilon(), None) + 1.), [1, 2, 3, 4]))

        decoded_art2ref = Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_art2ref._keras_shape)(
            decoded_art2ref)
        a_art2ref = K.square(
            decoded_art2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1, :] - decoded_art2ref[:, :, 1:, :patchSize[1] - 1, :])
        b_art2ref = K.square(
            decoded_art2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1, :] - decoded_art2ref[:, :, :patchSize[0] - 1, 1:, :])
        # (128, 1, 47, 47)
        gradient_art2ref = K.sqrt(a_art2ref + b_art2ref + K.epsilon())
        # (128,)
        sum_gradient_art2ref = K.sum(gradient_art2ref, [1, 2, 3, 4])
        # (128, 1, 1, 1)
        sum_gradient_art2ref = K.reshape(sum_gradient_art2ref, shape=(gradient_art2ref.shape[0], 1, 1, 1, 1))
        # (128, 1, 47, 47)
        h_art2ref = gradient_art2ref / sum_gradient_art2ref
        # (1,)
        ge_art2ref = K.mean(K.sum(-h_art2ref * K.log(K.clip(h_art2ref, K.epsilon(), None) + 1.), [1, 2, 3, 4]))

    return ge_ref2ref, ge_art2ref


def compute_tv_loss(dHyper, decoded_ref2ref, decoded_art2ref, patchSize):
    if K.ndim(decoded_ref2ref) == 4 and K.ndim(decoded_art2ref) == 4:
        decoded_ref2ref = Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(
            decoded_ref2ref)
        a_ref2ref = K.square(
            decoded_ref2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1] - decoded_ref2ref[:, :, 1:, :patchSize[1] - 1])
        b_ref2ref = K.square(
            decoded_ref2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1] - decoded_ref2ref[:, :, :patchSize[0] - 1, 1:])
        tv_loss_ref2ref = K.sum(K.pow(a_ref2ref + b_ref2ref, 1.25))

        decoded_art2ref = Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_art2ref._keras_shape)(
            decoded_art2ref)
        a_art2ref = K.square(
            decoded_art2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1] - decoded_art2ref[:, :, 1:, :patchSize[1] - 1])
        b_art2ref = K.square(
            decoded_art2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1] - decoded_art2ref[:, :, :patchSize[0] - 1, 1:])
        tv_loss_art2ref = K.sum(K.pow(a_art2ref + b_art2ref, 1.25))

    elif K.ndim(decoded_ref2ref) == 5 and K.ndim(decoded_art2ref) == 5:
        decoded_ref2ref = Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_ref2ref._keras_shape)(
            decoded_ref2ref)
        a_ref2ref = K.square(decoded_ref2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1, :] - decoded_ref2ref[:, :, 1:,
                                                                                              :patchSize[1] - 1, :])
        b_ref2ref = K.square(
            decoded_ref2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1, :] - decoded_ref2ref[:, :, :patchSize[0] - 1,
                                                                             1:, :])
        tv_loss_ref2ref = K.sum(K.pow(a_ref2ref + b_ref2ref, 1.25))

        decoded_art2ref = Lambda(lambda x: dHyper['nScale'] * x, output_shape=decoded_art2ref._keras_shape)(
            decoded_art2ref)
        a_art2ref = K.square(decoded_art2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1, :] - decoded_art2ref[:, :, 1:,
                                                                                              :patchSize[1] - 1, :])
        b_art2ref = K.square(
            decoded_art2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1, :] - decoded_art2ref[:, :, :patchSize[0] - 1,
                                                                             1:, :])
        tv_loss_art2ref = K.sum(K.pow(a_art2ref + b_art2ref, 1.25))

    return tv_loss_ref2ref, tv_loss_art2ref


def compute_perceptual_loss(x_ref, decoded_ref2ref, decoded_art2ref, patchSize, pl_network, loss_model):
    # if K.ndim(x_ref) == 5 and K.ndim(decoded_ref2ref) == 5 and K.ndim(decoded_art2ref) == 5:
    #     x_ref = reshape(x_ref, patchSize)
    #     decoded_ref2ref = reshape(decoded_ref2ref, patchSize)
    #     decoded_art2ref = reshape(decoded_art2ref, patchSize)

    # if K.ndim(x_ref) == 5 and K.ndim(decoded_ref2ref) == 5 and K.ndim(decoded_art2ref) == 5:
        # x_ref = K.mean(x_ref, axis=-1)
        # decoded_ref2ref = K.mean(decoded_ref2ref, axis=-1)
        # decoded_art2ref = K.mean(decoded_art2ref, axis=-1)

    #     x_ref = reshape(x_ref, patchSize)
    #     decoded_ref2ref = reshape(decoded_ref2ref, [20, 1, patchSize[0], patchSize[1]])
    #     decoded_art2ref = reshape(decoded_art2ref, [20, 1, patchSize[0], patchSize[1]])

    if pl_network == 'vgg19':
        if len(patchSize) == 2:
            x_ref = concatenate([x_ref, x_ref, x_ref], axis=1)
            decoded_ref2ref = concatenate([decoded_ref2ref, decoded_ref2ref, decoded_ref2ref], axis=1)
            decoded_art2ref = concatenate([decoded_art2ref, decoded_art2ref, decoded_art2ref], axis=1)

            x_ref = Lambda(preprocessing, output_shape=(3, patchSize[0], patchSize[1]))(x_ref)
            decoded_ref2ref = Lambda(preprocessing, output_shape=(3, patchSize[0], patchSize[1]))(decoded_ref2ref)
            decoded_art2ref = Lambda(preprocessing, output_shape=(3, patchSize[0], patchSize[1]))(decoded_art2ref)

            input = Input(shape=(3, patchSize[0], patchSize[1]))

        elif len(patchSize) == 3:
            x_ref = Lambda(lambda x: K.mean(x, axis=4), output_shape=(1, patchSize[0], patchSize[1]))(x_ref)
            decoded_art2ref = Lambda(lambda x: K.mean(x, axis=4), output_shape=(1, patchSize[0], patchSize[1]))(decoded_art2ref)
            decoded_ref2ref = Lambda(lambda x: K.mean(x, axis=4), output_shape=(1, patchSize[0], patchSize[1]))(decoded_ref2ref)

            x_ref = concatenate([x_ref, x_ref, x_ref], axis=1)
            decoded_ref2ref = concatenate([decoded_ref2ref, decoded_ref2ref, decoded_ref2ref], axis=1)
            decoded_art2ref = concatenate([decoded_art2ref, decoded_art2ref, decoded_art2ref], axis=1)

            x_ref = Lambda(preprocessing, output_shape=(3, patchSize[0], patchSize[1]))(x_ref)
            decoded_ref2ref = Lambda(preprocessing, output_shape=(3, patchSize[0], patchSize[1]))(decoded_ref2ref)
            decoded_art2ref = Lambda(preprocessing, output_shape=(3, patchSize[0], patchSize[1]))(decoded_art2ref)

            input = Input(shape=(3, patchSize[0], patchSize[1]))

        model = VGG19(include_top=False, weights='imagenet', input_tensor=input)

    elif pl_network == 'motion_head':
        model = load_model(loss_model)
        input = model.input

    else:
        sys.exit("loss network is not supported.")

    l1 = model.layers[1].output
    l2 = model.layers[4].output
    l3 = model.layers[7].output

    l1_model = Model(input, l1)
    l2_model = Model(input, l2)
    l3_model = Model(input, l3)

    f_l1_ref = l1_model(x_ref)
    f_l2_ref = l2_model(x_ref)
    f_l3_ref = l3_model(x_ref)

    f_l1_decoded_ref = l1_model(decoded_ref2ref)
    f_l2_decoded_ref = l2_model(decoded_ref2ref)
    f_l3_decoded_ref = l3_model(decoded_ref2ref)
    f_l1_decoded_art = l1_model(decoded_art2ref)
    f_l2_decoded_art = l2_model(decoded_art2ref)
    f_l3_decoded_art = l3_model(decoded_art2ref)

    if len(patchSize) == 2:
        p1_loss_ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))(
            [f_l1_ref, f_l1_decoded_ref])
        p2_loss_ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))(
            [f_l2_ref, f_l2_decoded_ref])
        p3_loss_ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))(
            [f_l3_ref, f_l3_decoded_ref])

        p1_loss_art = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))(
            [f_l1_ref, f_l1_decoded_art])
        p2_loss_art = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))(
            [f_l2_ref, f_l2_decoded_art])
        p3_loss_art = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))(
            [f_l3_ref, f_l3_decoded_art])

    elif len(patchSize) == 3:
        p1_loss_ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))(
            [f_l1_ref, f_l1_decoded_ref])
        p2_loss_ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))(
            [f_l2_ref, f_l2_decoded_ref])
        p3_loss_ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))(
            [f_l3_ref, f_l3_decoded_ref])

        p1_loss_art = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))(
            [f_l1_ref, f_l1_decoded_art])
        p2_loss_art = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))(
            [f_l2_ref, f_l2_decoded_art])
        p3_loss_art = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))(
            [f_l3_ref, f_l3_decoded_art])

    perceptual_loss_ref2ref = p1_loss_ref + p2_loss_ref + p3_loss_ref
    perceptual_loss_art2ref = p1_loss_art + p2_loss_art + p3_loss_art

    return perceptual_loss_ref2ref, perceptual_loss_art2ref
