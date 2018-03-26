import sys
from keras.layers import concatenate, Lambda, Input
from keras.metrics import mse
from keras import backend as K
from keras.applications.vgg19 import VGG19
from keras.models import Model, load_model


def preprocessing(inputs):
    output = Lambda(lambda x: (x - K.min(x)) * 255 / (K.max(x) - K.min(x)), output_shape=inputs._keras_shape)(inputs)
    # output = inputs * 255
    K.update_sub(output[:, 0, :, :], 123.68)
    K.update_sub(output[:, 1, :, :], 116.779)
    K.update_sub(output[:, 2, :, :], 103.939)

    return output[:, ::-1, :, :]


def reshape(inputs, patchSize):
    return K.reshape(inputs, (-1, 1, patchSize[0], patchSize[1]))


def compute_mse_loss(dHyper, x_ref, decoded_ref2ref, decoded_art2ref):
    loss_ref2ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,)) \
                       ([Lambda(lambda x: (x - K.min(x)) * dHyper['nScale'] / (K.max(x) - K.min(x)), output_shape=x_ref._keras_shape)(x_ref),
                         Lambda(lambda x: (x - K.min(x)) * dHyper['nScale'] / (K.max(x) - K.min(x)), output_shape=decoded_ref2ref._keras_shape)(decoded_ref2ref)])

    loss_art2ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))\
                       ([Lambda(lambda x: (x - K.min(x)) * dHyper['nScale'] / (K.max(x) - K.min(x)), output_shape=x_ref._keras_shape)(x_ref),
                         Lambda(lambda x: (x - K.min(x)) * dHyper['nScale'] / (K.max(x) - K.min(x)), output_shape=decoded_art2ref._keras_shape)(decoded_art2ref)])

    return loss_ref2ref, loss_art2ref


def compute_tv_loss(dHyper, decoded_ref2ref, decoded_art2ref, patchSize):
    if K.ndim(decoded_ref2ref) == 4 and K.ndim(decoded_art2ref) == 4:
        decoded_ref2ref = Lambda(lambda x: (x - K.min(x)) * dHyper['nScale'] / (K.max(x) - K.min(x)), output_shape=decoded_ref2ref._keras_shape)(decoded_ref2ref)
        a_ref2ref = K.square(decoded_ref2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1] - decoded_ref2ref[:, :, 1:, :patchSize[1] - 1])
        b_ref2ref = K.square(decoded_ref2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1] - decoded_ref2ref[:, :, :patchSize[0] - 1, 1:])
        tv_loss_ref2ref = K.sum(K.pow(a_ref2ref + b_ref2ref, 1.25))

        decoded_art2ref = Lambda(lambda x: (x - K.min(x)) * dHyper['nScale'] / (K.max(x) - K.min(x)), output_shape=decoded_art2ref._keras_shape)(decoded_art2ref)
        a_art2ref = K.square(decoded_art2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1] - decoded_art2ref[:, :, 1:, :patchSize[1] - 1])
        b_art2ref = K.square(decoded_art2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1] - decoded_art2ref[:, :, :patchSize[0] - 1, 1:])
        tv_loss_art2ref = K.sum(K.pow(a_art2ref + b_art2ref, 1.25))

    elif K.ndim(decoded_ref2ref) == 5 and K.ndim(decoded_art2ref) == 5:
        decoded_ref2ref = Lambda(lambda x: (x - K.min(x)) * dHyper['nScale'] / (K.max(x) - K.min(x)), output_shape=decoded_ref2ref._keras_shape)(decoded_ref2ref)
        a_ref2ref = K.square(decoded_ref2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1, :] - decoded_ref2ref[:, :, 1:, :patchSize[1] - 1, :])
        b_ref2ref = K.square(decoded_ref2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1, :] - decoded_ref2ref[:, :, :patchSize[0] - 1, 1:, :])
        tv_loss_ref2ref = K.sum(K.pow(a_ref2ref + b_ref2ref, 1.25))

        decoded_art2ref = Lambda(lambda x: (x - K.min(x)) * dHyper['nScale'] / (K.max(x) - K.min(x)), output_shape=decoded_art2ref._keras_shape)(decoded_art2ref)
        a_art2ref = K.square(decoded_art2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1, :] - decoded_art2ref[:, :, 1:, :patchSize[1] - 1, :])
        b_art2ref = K.square(decoded_art2ref[:, :, :patchSize[0] - 1, :patchSize[1] - 1, :] - decoded_art2ref[:, :, :patchSize[0] - 1, 1:, :])
        tv_loss_art2ref = K.sum(K.pow(a_art2ref + b_art2ref, 1.25))

    return tv_loss_ref2ref, tv_loss_art2ref


def compute_perceptual_loss(x_ref, decoded_ref2ref, decoded_art2ref, patchSize, pl_network, loss_model):
    if K.ndim(x_ref) == 5 and K.ndim(decoded_ref2ref) == 5 and K.ndim(decoded_art2ref) == 5:
        x_ref = reshape(x_ref, patchSize)
        decoded_ref2ref = reshape(decoded_ref2ref, patchSize)
        decoded_art2ref = reshape(decoded_art2ref, patchSize)

    if pl_network == 'vgg19':
        x_ref = concatenate([x_ref, x_ref, x_ref], axis=1)
        decoded_ref2ref = concatenate([decoded_ref2ref, decoded_ref2ref, decoded_ref2ref], axis=1)
        decoded_art2ref = concatenate([decoded_art2ref, decoded_art2ref, decoded_art2ref], axis=1)

        # x_ref = Lambda(preprocessing, output_shape=(3, patchSize[0], patchSize[1]))(x_ref)
        # decoded_ref2ref = Lambda(preprocessing, output_shape=(3, patchSize[0], patchSize[1]))(decoded_ref2ref)
        # decoded_art2ref = Lambda(preprocessing, output_shape=(3, patchSize[0], patchSize[1]))(decoded_art2ref)

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

    p1_loss_ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))([f_l1_ref, f_l1_decoded_ref])
    p2_loss_ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))([f_l2_ref, f_l2_decoded_ref])
    p3_loss_ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))([f_l3_ref, f_l3_decoded_ref])

    p1_loss_art = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))([f_l1_ref, f_l1_decoded_art])
    p2_loss_art = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))([f_l2_ref, f_l2_decoded_art])
    p3_loss_art = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])), output_shape=(None,))([f_l3_ref, f_l3_decoded_art])

    perceptual_loss_ref2ref = p1_loss_ref + p2_loss_ref + p3_loss_ref
    perceptual_loss_art2ref = p1_loss_art + p2_loss_art + p3_loss_art

    return perceptual_loss_ref2ref, perceptual_loss_art2ref
