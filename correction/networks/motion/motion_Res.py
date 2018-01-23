import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Lambda, Layer, concatenate, LeakyReLU, GaussianNoise
from keras.layers import Conv2D, Conv2DTranspose, add
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras import metrics
from keras.optimizers import Adam
import os

# Custom loss layer
class LossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(LossLayer, self).__init__(**kwargs)

    def vae_loss(self, x_ref, x_decoded, shared):
        x_ref = K.flatten(x_ref)

        decoded_ref2ref = Lambda(sliceRef)(x_decoded)
        decoded_art2ref = Lambda(sliceArt)(x_decoded)
        decoded_ref2ref = K.flatten(decoded_ref2ref)
        decoded_art2ref = K.flatten(decoded_art2ref)

        loss_ref2ref =  1600 * metrics.binary_crossentropy(x_ref, decoded_ref2ref)
        loss_art2ref = 1600 * metrics.binary_crossentropy(x_ref, decoded_art2ref)

        loss_kl = K.mean(K.pow(shared, 2))
        return 100 * (loss_ref2ref + loss_art2ref) + 0.1 * loss_kl

    def call(self, inputs):
        x_ref = inputs[0]
        x_decoded = inputs[1]
        shared = inputs[2]
        loss = self.vae_loss(x_ref, x_decoded, shared)
        self.add_loss(loss)
        return x_decoded

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

def ReluResBlock(filters, kernel_size, strides, padding):
    def f(inputs):
        conv_1 = Conv2D(filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        activation='relu')(inputs)

        conv_2 = Conv2D(filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        activation='relu')(conv_1)
        return add([inputs, conv_2])
    return f

def sliceRef(input):
    return input[:input.shape[0]//2, :, :, :]

def sliceArt(input):
    return input[input.shape[0]//2:, :, :, :]

def decode(input):
    output = LeakyReluConv2DTranspose(filters=256, kernel_size=(2, 2), strides=2, padding='valid')(input)
    output = LeakyReluConv2DTranspose(filters=128, kernel_size=(3, 3), strides=1, padding='valid')(output)
    output = LeakyReluConv2DTranspose(filters=64, kernel_size=(3, 3), strides=1, padding='valid')(output)
    output = LeakyReluConv2DTranspose(filters=1, kernel_size=(3, 3), strides=1, padding='valid')(output)
    return output

def encode(input):
    output = LeakyReluConv2D(filters=64, kernel_size=(3, 3), strides=1, padding='valid')(input)
    output = LeakyReluConv2D(filters=128, kernel_size=(3, 3), strides=1, padding='valid')(output)
    output = LeakyReluConv2D(filters=256, kernel_size=(3, 3), strides=2, padding='valid')(output)
    return output

def encode_shared(input):
    output = ReluResBlock(filters=256, kernel_size=(3, 3), strides=1, padding='same')(input)
    output = ReluResBlock(filters=256, kernel_size=(3, 3), strides=1, padding='same')(output)
    output = ReluResBlock(filters=256, kernel_size=(3, 3), strides=1, padding='same')(output)
    output = ReluResBlock(filters=256, kernel_size=(3, 3), strides=1, padding='same')(output)
    output = GaussianNoise(stddev=1.0)(output)
    return output

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
    output = LossLayer()([x_ref, decoded, shared])

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
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    vae.compile(optimizer=adam, loss=None)
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
    predict_ref, predict_art = vae.predict([dData['test_ref'][:128], dData['test_art'][:128]], 128, verbose=1)

    test_ref = np.squeeze(dData['test_ref'][:128], axis=1)
    test_art = np.squeeze(dData['test_art'][:128], axis=1)
    predict_ref = np.squeeze(predict_ref, axis=1)
    predict_art = np.squeeze(predict_art, axis=1)

    nPatch = predict_ref.shape[0]

    fig = plt.figure()
    plt.gray()

    for i in range(1, 6):
        iPatch = np.random.randint(0, nPatch)

        fig.add_subplot(5, 2, 2*i-1)
        plt.imshow(test_ref[iPatch])

        fig.add_subplot(5, 2, 2*i)
        plt.imshow(predict_ref[iPatch])

    plt.show()

    fig = plt.figure()

    for i in range(1, 6):
        iPatch = np.random.randint(0, nPatch)

        fig.add_subplot(5, 2, 2*i-1)
        plt.imshow(test_art[iPatch])

        fig.add_subplot(5, 2, 2*i)
        plt.imshow(predict_art[iPatch])

    plt.show()

