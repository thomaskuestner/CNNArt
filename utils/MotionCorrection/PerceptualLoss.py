import sys
from keras.layers import concatenate, Lambda, Input
from keras import backend as K
from keras.applications.vgg19 import VGG19
from keras.models import Model, load_model

def preprocessing(input):
    output = input * 255
    K.update_sub(output[:, 0, :, :], 123.68)
    K.update_sub(output[:, 1, :, :], 116.779)
    K.update_sub(output[:, 2, :, :], 103.939)

    return output[:, ::-1, :, :]

def addPerceptualLoss(x_ref, decoded_ref2ref, decoded_art2ref, patchSize, perceptual_weight, pl_network, loss_model):
    if pl_network == 'vgg19':
        x_ref = concatenate([x_ref, x_ref, x_ref], axis=1)
        decoded_ref2ref = concatenate([decoded_ref2ref, decoded_ref2ref, decoded_ref2ref], axis=1)
        decoded_art2ref = concatenate([decoded_art2ref, decoded_art2ref, decoded_art2ref], axis=1)

        x_ref = Lambda(preprocessing)(x_ref)
        decoded_ref2ref = Lambda(preprocessing)(decoded_ref2ref)
        decoded_art2ref = Lambda(preprocessing)(decoded_art2ref)

        input = Input(shape=(3, patchSize[0], patchSize[1]))

        model = VGG19(include_top=False, weights='imagenet', input_tensor=input)

    # TODO: adapt the path
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

    p1_loss_ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])))([f_l1_ref, f_l1_decoded_ref])
    p2_loss_ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])))([f_l2_ref, f_l2_decoded_ref])
    p3_loss_ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])))([f_l3_ref, f_l3_decoded_ref])

    p1_loss_art = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])))([f_l1_ref, f_l1_decoded_art])
    p2_loss_art = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])))([f_l2_ref, f_l2_decoded_art])
    p3_loss_art = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])))([f_l3_ref, f_l3_decoded_art])

    p_loss = perceptual_weight * (p1_loss_ref + p2_loss_ref + p3_loss_ref + p1_loss_art + p2_loss_art + p3_loss_art)

    return p_loss
