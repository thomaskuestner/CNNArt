import sys
from keras.layers import concatenate, Lambda, Input
from keras import backend as K
from keras.applications.vgg19 import VGG19
from keras.models import Model

def preprocessing(input):
    output = input * 255
    K.update_sub(output[:, 0, :, :], 123.68)
    K.update_sub(output[:, 1, :, :], 116.779)
    K.update_sub(output[:, 2, :, :], 103.939)

    return output[:, ::-1, :, :]

def addPerceptualLoss(x_ref, decoded_ref2ref, decoded_art2ref, patchSize, perceptual_weight, pl_network):
    if pl_network == 'vgg19':
        x_ref_triple = concatenate([x_ref, x_ref, x_ref], axis=1)
        decoded_ref_triple = concatenate([decoded_ref2ref, decoded_ref2ref, decoded_ref2ref], axis=1)
        decoded_art_triple = concatenate([decoded_art2ref, decoded_art2ref, decoded_art2ref], axis=1)

        x_ref_triple = Lambda(preprocessing)(x_ref_triple)
        decoded_ref_triple = Lambda(preprocessing)(decoded_ref_triple)
        decoded_art_triple = Lambda(preprocessing)(decoded_art_triple)

        vgg_input = Input(shape=(3, patchSize[0], patchSize[1]))

        vgg = VGG19(include_top=False, weights='imagenet', input_tensor=vgg_input)
        vgg.trainable = False
        for l in vgg.layers:
            l.trainable = False

        l1 = vgg.layers[1].output
        l2 = vgg.layers[4].output
        l3 = vgg.layers[7].output

        # making model Model(inputs, outputs)
        l1_model = Model(vgg_input, l1)
        l2_model = Model(vgg_input, l2)
        l3_model = Model(vgg_input, l3)

        l1_model.trainable = False
        l2_model.trainable = False
        l3_model.trainable = False
        for l in l1_model.layers:
            l.trainable = False
        for l in l2_model.layers:
            l.trainable = False
        for l in l3_model.layers:
            l.trainable = False

        f_l1_ref = l1_model(x_ref_triple)
        f_l2_ref = l2_model(x_ref_triple)
        f_l3_ref = l3_model(x_ref_triple)
        f_l1_art = l1_model(decoded_art_triple)
        f_l2_art = l2_model(decoded_art_triple)
        f_l3_art = l3_model(decoded_art_triple)
        f_l1_predict = l1_model(decoded_ref_triple)
        f_l2_predict = l2_model(decoded_ref_triple)
        f_l3_predict = l3_model(decoded_ref_triple)

        p1_loss_ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])))([f_l1_ref, f_l1_predict])
        p2_loss_ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])))([f_l2_ref, f_l2_predict])
        p3_loss_ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])))([f_l3_ref, f_l3_predict])

        p1_loss_art = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])))([f_l1_art, f_l1_predict])
        p2_loss_art = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])))([f_l2_art, f_l2_predict])
        p3_loss_art = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3])))([f_l3_art, f_l3_predict])

        p_loss = perceptual_weight * (p1_loss_ref + p2_loss_ref + p3_loss_ref + p1_loss_art + p2_loss_art + p3_loss_art)

    else:
        sys.exit("loss network is not supported.")

    return p_loss
