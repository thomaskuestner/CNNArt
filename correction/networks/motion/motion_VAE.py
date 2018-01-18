import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer, concatenate
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras import metrics
import os

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        self.patch_size = [40, 40]
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x_ref, x_art, x_decoded_mean_squash, z_mean, z_log_var):
        x_ref = K.flatten(x_ref)
        x_art = K.flatten(x_art)

        decoded_ref2ref = Lambda(sliceRef)(x_decoded_mean_squash)
        decoded_art2ref = Lambda(sliceArt)(x_decoded_mean_squash)
        decoded_ref2ref = K.flatten(decoded_ref2ref)
        decoded_art2ref = K.flatten(decoded_art2ref)

        # multiply patch size?
        loss_ref2ref = self.patch_size[0] * self.patch_size[1] * metrics.binary_crossentropy(x_ref, decoded_ref2ref)
        loss_art2ref = self.patch_size[0] * self.patch_size[1] * metrics.binary_crossentropy(x_art, decoded_art2ref)

        # TODO: add kl loss
        # kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(loss_ref2ref + loss_art2ref)

    def call(self, inputs):
        x_ref = inputs[0]
        x_art = inputs[1]
        x_decoded_mean_squash = inputs[2]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        loss = self.vae_loss(x_ref, x_art, x_decoded_mean_squash, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        return x_decoded_mean_squash

def sliceRef(input):
    return input[:input.shape[0]//2, :, :, :]

def sliceArt(input):
    return input[input.shape[0]//2:, :, :, :]

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 2),
                              mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var) * epsilon

def createDecoder(z):
    # we instantiate these layers separately so as to reuse them later
    hid_decoder = Dense(128, activation='relu')(z)
    up_decoded = Dense(64 * 20 * 20, activation='relu')(hid_decoder)
    reshape_decoded = Reshape((64, 20, 20))(up_decoded)
    deconv_1_decoded = Conv2DTranspose(64,
                                       kernel_size=(3, 3),
                                       padding='same',
                                       strides=1,
                                       activation='relu')(reshape_decoded)
    deconv_2_decoded = Conv2DTranspose(64,
                                       kernel_size=(3, 3),
                                       padding='same',
                                       strides=1,
                                       activation='relu')(deconv_1_decoded)
    x_decoded_relu = Conv2DTranspose(64,
                                     kernel_size=(3, 3),
                                     strides=(2, 2),
                                     padding='valid',
                                     activation='relu')(deconv_2_decoded)
    x_decoded_mean_squash = Conv2D(1,
                                   kernel_size=2,
                                   padding='valid',
                                   activation='sigmoid')(x_decoded_relu)

    return x_decoded_mean_squash

def createEncoder(x):
    conv_1 = Conv2D(1,
                    kernel_size=(2, 2),
                    padding='same', activation='relu')(x)
    conv_2 = Conv2D(64,
                    kernel_size=(2, 2),
                    padding='same', activation='relu',
                    strides=(2, 2))(conv_1)
    conv_3 = Conv2D(64,
                    kernel_size=(3, 3),
                    padding='same', activation='relu',
                    strides=1)(conv_2)
    conv_4 = Conv2D(64,
                    kernel_size=(3, 3),
                    padding='same', activation='relu',
                    strides=1)(conv_3)
    return conv_4

def createSharedEncoder(input):
    flat = Flatten()(input)
    hidden = Dense(128, activation='relu')(flat)

    z_mean = Dense(2)(hidden)
    z_log_var = Dense(2)(hidden)

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z = Lambda(sampling, output_shape=(2,))([z_mean, z_log_var])

    return z, z_mean, z_log_var

def createModel(patchSize):
    # input corrupted and non-corrupted image
    x_ref = Input(shape=(1, patchSize[0], patchSize[1]))
    x_art = Input(shape=(1, patchSize[0], patchSize[1]))

    # create respective encoder
    encoded_ref = createEncoder(x_ref)
    encoded_art = createEncoder(x_art)

    # concatenate the encoded features together
    combined = concatenate([encoded_ref, encoded_art], axis=0)

    # create the shared encoder
    z, z_mean, z_log_var = createSharedEncoder(combined)

    # create the decoder
    decoded = createDecoder(z)

    # create a customer layer to calculate the total loss
    output = CustomVariationalLayer()([x_ref, x_art, decoded, z_mean, z_log_var])

    decoded_ref2ref = Lambda(sliceRef)(output)
    decoded_art2ref = Lambda(sliceArt)(output)

    # generate the VAE and encoder model
    vae = Model([x_ref, x_art], [decoded_ref2ref, decoded_art2ref])
    encoder = Model([x_ref, x_art], z_mean)
    return encoder, vae

def fTrain(dData, sOutPath, patchSize, dHyper):
    # parse inputs
    batchSize = [128] if dHyper['batchSize'] is None else dHyper['batchSize']
    learningRate = [0.001] if dHyper['learningRate'] is None else dHyper['learningRate']
    epochs = 300 if dHyper['epochs'] is None else dHyper['epochs']

    for iBatch in batchSize:
        for iLearn in learningRate:
            fTrainInner(dData, sOutPath, patchSize, epochs, iBatch, iLearn)

def fTrainInner(dData, sOutPath, patchSize, epochs, batchSize, lr):
    train_ref = dData['train_ref']
    train_art = dData['train_art']
    test_ref = dData['test_ref']
    test_art = dData['test_art']

    train_ref = np.expand_dims(train_ref, axis=1)
    train_art = np.expand_dims(train_art, axis=1)
    test_ref = np.expand_dims(test_ref, axis=1)
    test_art = np.expand_dims(test_art, axis=1)

    encoder, vae = createModel(patchSize)
    vae.compile(optimizer='adam', loss=None)
    vae.summary()

    print('Training with epochs {} batch size {} learning rate {}'.format(epochs, batchSize, lr))

    weights_file = sOutPath + os.sep + 'vae_model_weight_bs_{}.h5'.format(batchSize)

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

    encoder, vae = createModel(patchSize)
    vae.compile(optimizer='adam', loss=None)

    vae.load_weights(weights_file)

    # display a 2D plot of the digit classes in the latent space
    # x_test__ref_encoded = encoder.predict(dData['test_ref'], 128)
    # x_test__arf_encoded = encoder.predict(dData['test_art'], 128)
    # plt.figure()
    # plt.scatter(x_test__ref_encoded[:, 0], x_test__ref_encoded[:, 1], c='r')
    # plt.scatter(x_test__arf_encoded[:, 0], x_test__arf_encoded[:, 1], c='b')
    # plt.show()

    # TODO: adapt the embedded batch size
    predict_ref, predict_art = vae.predict([dData['test_ref'], dData['test_art']], 128, verbose=1)

    test_ref = np.squeeze(dData['test_ref'], axis=1)
    test_art = np.squeeze(dData['test_art'], axis=1)
    predict_ref = np.squeeze(predict_ref, axis=1)
    predict_art = np.squeeze(predict_art, axis=1)

    fig = plt.figure()
    plt.gray()

    fig.add_subplot(2, 2, 1)
    plt.imshow(test_ref[50])

    fig.add_subplot(2, 2, 2)
    plt.imshow(predict_ref[50])

    fig.add_subplot(2, 2, 3)
    plt.imshow(test_art[50])

    fig.add_subplot(2, 2, 4)
    plt.imshow(predict_art[50])

    plt.show()





