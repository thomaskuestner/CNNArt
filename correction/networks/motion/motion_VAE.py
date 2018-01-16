import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras import metrics
import os
import glob

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash, z_mean, z_log_var):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = 180 * 180 * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        loss = self.vae_loss(x, x_decoded_mean_squash, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 2),
                              mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var) * epsilon

def createModel(patchSize):
    x = Input(shape= (1, patchSize[0], patchSize[1]))
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
    flat = Flatten()(conv_4)
    hidden = Dense(128, activation='relu')(flat)

    z_mean = Dense(2)(hidden)
    z_log_var = Dense(2)(hidden)

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z = Lambda(sampling, output_shape=(2,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    hid_decoder = Dense(128, activation='relu')(z)
    up_decoded = Dense(64 * 90 * 90, activation='relu')(hid_decoder)
    reshape_decoded = Reshape((64, 90, 90))(up_decoded)
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

    y = CustomVariationalLayer()([x, x_decoded_mean_squash, z_mean, z_log_var])
    vae = Model(x, y)
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

    weights_file = sOutPath + os.sep + 'vae_model_weight_ps_{}_bs_{}.h5'.format(patchSize[0], batchSize)

    callback_list = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
    callback_list.append(ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, period=1, save_best_only=True, save_weights_only=True))
    callback_list.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-4, verbose=1))

    vae.fit(train_ref,
            shuffle=True,
            epochs=epochs,
            batch_size=batchSize,
            validation_data=(test_ref, None),
            verbose=1,
            callbacks=callback_list)

def fPredict(dData, sOutPath, patchSize, dHyper):
    weights_file = sOutPath + os.sep + '{}.h5'.format(dHyper['bestModel'])

    vae = createModel(patchSize)
    vae.compile(optimizer='adam', loss=None)

    vae.load_weights(weights_file)

    # TODO: adapt the embedded batch size
    predict_ref = vae.predict(dData['test_ref'], 64, verbose=1)
    predict_art = vae.predict(dData['test_art'], 64, verbose=1)

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





