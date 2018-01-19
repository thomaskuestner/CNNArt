import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer, concatenate, LeakyReLU
from keras.layers import Conv2D, Conv2DTranspose, add
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras import metrics
import os

# Custom loss layer
class lossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        self.patch_size = [40, 40]

        super(lossLayer, self).__init__(**kwargs)

    def vae_loss(self, x_ref, x_art, x_decoded):
        x_ref = K.flatten(x_ref)
        x_art = K.flatten(x_art)

        decoded_ref2ref = Lambda(sliceRef)(x_decoded)
        decoded_art2ref = Lambda(sliceArt)(x_decoded)
        decoded_ref2ref = K.flatten(decoded_ref2ref)
        decoded_art2ref = K.flatten(decoded_art2ref)

        loss_ref2ref = self.patch_size[0] * self.patch_size[1] * metrics.binary_crossentropy(x_ref, decoded_ref2ref)
        loss_art2ref = self.patch_size[0] * self.patch_size[1] * metrics.binary_crossentropy(x_art, decoded_art2ref)

        # TODO: add kl loss
        return K.mean(loss_ref2ref + loss_art2ref)

    def call(self, inputs):
        x_ref = inputs[0]
        x_art = inputs[1]
        x_decoded = inputs[2]
        loss = self.vae_loss(x_ref, x_art, x_decoded)
        self.add_loss(loss, inputs=inputs)
        return x_decoded

def sliceRef(input):
    return input[:input.shape[0]//2, :, :, :]

def sliceArt(input):
    return input[input.shape[0]//2:, :, :, :]

def decode(shared):
    # Residual-block back-end
    conv_1 = Conv2D(256,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-6))(shared)
    conv_1_r = LeakyReLU()(conv_1)
    conv_1 = LeakyReLU()(conv_1)

    conv_2 = Conv2D(256,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-6))(conv_1)
    conv_2 = LeakyReLU()(conv_2)
    conv_2 = add([conv_1_r, conv_2])

    conv_3 = Conv2D(128,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-6))(conv_2)
    conv_3_r = LeakyReLU()(conv_3)
    conv_3 = LeakyReLU()(conv_3)

    conv_4 = Conv2D(128,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-6))(conv_3)
    conv_4 = LeakyReLU()(conv_4)
    conv_4 = add([conv_3_r, conv_4])

    # Convolutional back-end
    deconv_5 = Conv2DTranspose(64,
                               kernel_size=(3, 3),
                               strides=(2, 2),
                               padding='valid',
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(1e-6))(conv_4)
    deconv_5 = LeakyReLU()(deconv_5)

    deconv_6 = Conv2DTranspose(64,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(1e-6))(deconv_5)
    deconv_6 = LeakyReLU()(deconv_6)

    deconv_7 = Conv2DTranspose(1,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding='valid',
                              kernel_initializer='he_normal',
                              kernel_regularizer=l2(1e-6))(deconv_6)
    deconv_7 = LeakyReLU()(deconv_7)

    decoded = Conv2D(1,
                     kernel_size=2,
                     padding='valid',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(1e-6),
                     activation='tanh')(deconv_7)

    return decoded

def encode(input):
    # Convolutional front-end
    conv_1 = Conv2D(64,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='valid',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-6))(input)
    conv_1 = LeakyReLU()(conv_1)

    conv_2 = Conv2D(64,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding='valid',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-6))(conv_1)
    conv_2 = LeakyReLU()(conv_2)

    # Residual-block back-end
    conv_3 = Conv2D(128,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-6))(conv_2)
    conv_3_r = LeakyReLU()(conv_3)
    conv_3 = LeakyReLU()(conv_3)

    conv_4 = Conv2D(128,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-6))(conv_3)
    conv_4 = LeakyReLU()(conv_4)
    conv_4 = add([conv_3_r, conv_4])

    conv_5 = Conv2D(256,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-6))(conv_4)
    conv_5_r = LeakyReLU()(conv_5)
    conv_5 = LeakyReLU()(conv_5)

    conv_6 = Conv2D(256,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-6))(conv_5)
    conv_6 = LeakyReLU()(conv_6)
    conv_6 = add([conv_5_r, conv_6])
    return conv_6

def encode_shared(input):
    conv_1 = Conv2D(256,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-6))(input)
    conv_1_r = LeakyReLU()(conv_1)
    conv_1 = LeakyReLU()(conv_1)

    conv_2 = Conv2D(256,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-6))(conv_1)
    conv_2 = LeakyReLU()(conv_2)
    conv_2 = add([conv_1_r, conv_2])

    conv_3 = Conv2D(256,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-6))(conv_2)
    conv_3_r = LeakyReLU()(conv_3)
    conv_3 = LeakyReLU()(conv_3)

    conv_4 = Conv2D(256,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-6))(conv_3)
    conv_4 = LeakyReLU()(conv_4)
    conv_4 = add([conv_3_r, conv_4])

    return conv_4

def createModel(patchSize):
    # input corrupted and non-corrupted image
    x_ref = Input(shape=(1, patchSize[0], patchSize[1]))
    x_art = Input(shape=(1, patchSize[0], patchSize[1]))

    # create respective encoders
    encoded_ref = encode(x_ref)
    encoded_art = encode(x_art)

    # concatenate the encoded features together
    combined = concatenate([encoded_ref, encoded_art], axis=0)

    # create the shared encoder
    shared = encode_shared(combined)

    # create the decoder
    decoded = decode(shared)

    # create a customer layer to calculate the total loss
    output = lossLayer()([x_ref, x_art, decoded])

    # separate the concatenated images
    decoded_ref2ref = Lambda(sliceRef)(output)
    decoded_art2ref = Lambda(sliceArt)(output)

    # generate the VAE and encoder model
    vae = Model([x_ref, x_art], [decoded_ref2ref, decoded_art2ref])
    return vae

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

    vae = createModel(patchSize)
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

    vae = createModel(patchSize)
    vae.compile(optimizer='adam', loss=None)

    vae.load_weights(weights_file)

    # TODO: adapt the embedded batch size
    predict_ref, predict_art = vae.predict([dData['test_ref'], dData['test_art']], 128, verbose=1)

    test_ref = np.squeeze(dData['test_ref'], axis=1)
    test_art = np.squeeze(dData['test_art'], axis=1)
    predict_ref = np.squeeze(predict_ref, axis=1)
    predict_art = np.squeeze(predict_art, axis=1)

    fig = plt.figure()
    plt.gray()

    nPatch = predict_ref.shape[0]

    for i in range(1, 6):
        iPatch = np.random.randint(0, nPatch)

        fig.add_subplot(5, 2, 2*i-1)
        plt.imshow(test_ref[iPatch])

        fig.add_subplot(5, 2, 2*i)
        plt.imshow(predict_ref[iPatch])

    plt.show()





