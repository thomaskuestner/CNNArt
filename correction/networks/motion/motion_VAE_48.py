import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Lambda, Layer, concatenate, LeakyReLU, Dense, Reshape, Flatten
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras import metrics
import os

def LeakyReluConv2D(filters, kernel_size, strides, padding):
    def f(inputs):
        conv2d = Conv2D(filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding)(inputs)
        return LeakyReLU()(conv2d)
    return f

def LeakyReluConv2DTranspose(filters, kernel_size, strides, padding):
    def f(inputs):
        conv2d = Conv2DTranspose(filters=filters,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding=padding)(inputs)
        return LeakyReLU()(conv2d)
    return f

def preprocessing(input):
    output = input * 255
    K.update_sub(output[:, 0, :, :], 123.68)
    K.update_sub(output[:, 1, :, :], 116.779)
    K.update_sub(output[:, 2, :, :], 103.939)

    return output[:, ::-1, :, :]

def sliceRef(input):
    return input[:input.shape[0]//2, :, :, :]

def sliceArt(input):
    return input[input.shape[0]//2:, :, :, :]

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 512), mean=0.,
                              stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def encode(input):
    conv_1 = LeakyReluConv2D(filters=32, kernel_size=3, strides=1, padding='same')(input)
    conv_2 = LeakyReluConv2D(filters=64, kernel_size=3, strides=2, padding='same')(conv_1)
    conv_3 = LeakyReluConv2D(filters=128, kernel_size=3, strides=1, padding='same')(conv_2)
    return conv_3

def encode_shared(input):
    conv_1 = LeakyReluConv2D(filters=256, kernel_size=3, strides=1, padding='same')(input)
    conv_2 = LeakyReluConv2D(filters=256, kernel_size=3, strides=2, padding='same')(conv_1)
    flat = Flatten()(conv_2)

    z_mean = Dense(512)(flat)
    z_log_var = Dense(512)(flat)

    z = Lambda(sampling, output_shape=(512,))([z_mean, z_log_var])

    return z, z_mean, z_log_var

def decode(input):
    dense = Dense(256 * 12 * 12)(input)
    reshape = Reshape((256, 12, 12))(dense)
    output = LeakyReluConv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same')(reshape)
    output = LeakyReluConv2DTranspose(filters=128, kernel_size=3, strides=1, padding='same')(output)
    output = LeakyReluConv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(output)
    output = Conv2DTranspose(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(output)
    return output

def createModel(patchSize, kl_weight, perceptual_weight):
    # input corrupted and non-corrupted image
    x_ref = Input(shape=(1, patchSize[0], patchSize[1]))
    x_art = Input(shape=(1, patchSize[0], patchSize[1]))

    # create respective encoders
    encoded_ref = encode(x_ref)
    encoded_art = encode(x_art)

    # concatenate the encoded features together
    combined = concatenate([encoded_ref, encoded_art], axis=0)

    # create the shared encoder
    z, z_mean, z_log_var = encode_shared(combined)

    # create the decoder
    decoded = decode(z)

    # separate the concatenated images
    decoded_ref2ref = Lambda(sliceRef)(decoded)
    decoded_art2ref = Lambda(sliceArt)(decoded)

    # generate the VAE and encoder model
    vae = Model([x_ref, x_art], [decoded_ref2ref, decoded_art2ref])

    # compute kl loss
    loss_kl = - 0.5 * K.sum(1 + z_mean - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae.add_loss(kl_weight * K.mean(loss_kl))

    # compute pixel to pixel loss
    # loss_ref2ref = patchSize[0] * patchSize[1] * metrics.binary_crossentropy(K.flatten(x_ref), K.flatten(decoded_ref2ref))
    # loss_art2ref = patchSize[0] * patchSize[1] * metrics.binary_crossentropy(K.flatten(x_ref), K.flatten(decoded_art2ref))

    # vae.add_loss(K.mean(loss_ref2ref + loss_art2ref + loss_kl))

    # add perceptual loss
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

    vae.add_loss(p_loss)

    return vae

def fTrain(dData, sOutPath, patchSize, dHyper):
    # parse inputs
    batchSize = [128] if dHyper['batchSize'] is None else dHyper['batchSize']
    learningRate = [0.001] if dHyper['learningRate'] is None else dHyper['learningRate']
    epochs = 300 if dHyper['epochs'] is None else dHyper['epochs']

    for iBatch in batchSize:
        for iLearn in learningRate:
            fTrainInner(dData, sOutPath, patchSize, epochs, iBatch, iLearn, dHyper['kl_weight'],
                        dHyper['perceptual_weight'])

def fTrainInner(dData, sOutPath, patchSize, epochs, batchSize, lr, kl_weight, perceptual_weight):
    train_ref = dData['train_ref']
    train_art = dData['train_art']
    test_ref = dData['test_ref']
    test_art = dData['test_art']

    train_ref = np.expand_dims(train_ref, axis=1)
    train_art = np.expand_dims(train_art, axis=1)
    test_ref = np.expand_dims(test_ref, axis=1)
    test_art = np.expand_dims(test_art, axis=1)

    vae = createModel(patchSize, kl_weight, perceptual_weight)
    vae.compile(optimizer='adam', loss=None)
    vae.summary()

    print('Training with epochs {} batch size {} learning rate {}'.format(epochs, batchSize, lr))

    weights_file = sOutPath + os.sep + 'vae_weight_ps_{}_bs_{}.h5'.format(patchSize[0], batchSize)

    callback_list = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
    callback_list.append(ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, period=1, save_best_only=True, save_weights_only=True))
    callback_list.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-4, verbose=1))

    vae.fit([train_ref, train_art],
            shuffle=True,
            epochs=epochs,
            batch_size=batchSize,
            validation_data=([test_ref, test_art], None),
            verbose=1,
            callbacks=callback_list)

def fPredict(dData, sOutPath, patchSize, dHyper):
    weights_file = sOutPath + os.sep + '{}.h5'.format(dHyper['bestModel'])

    vae = createModel(patchSize)
    vae.compile(optimizer='adam', loss=None)

    vae.load_weights(weights_file)

    # TODO: adapt the embedded batch size
    predict_ref, predict_art = vae.predict([dData['test_ref'][:128], dData['test_art'][:128]], 128, verbose=1)

    test_ref = np.squeeze(dData['test_ref'][:128], axis=1)
    test_art = np.squeeze(dData['test_art'][:128], axis=1)
    predict_ref = np.squeeze(predict_ref, axis=1)
    predict_art = np.squeeze(predict_art, axis=1)

    nPatch = predict_ref.shape[0]

    for i in range(nPatch//6):
        fig, axes = plt.subplots(nrows=5, ncols=4)
        plt.gray()

        cols_title = ['test_ref', 'predict_ref', 'test_art', 'predict_art']

        for ax, col in zip(axes[0], cols_title):
            ax.set_title(col)

        for j in range(5):
            axes[j, 0].imshow(test_ref[6*i+j])
            axes[j, 1].imshow(predict_ref[6 * i + j])
            axes[j, 2].imshow(test_art[6 * i + j])
            axes[j, 3].imshow(predict_art[6 * i + j])

        plt.show()


