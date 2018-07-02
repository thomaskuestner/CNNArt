import os
import datetime
from skimage.measure import compare_ssim as ssim
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
mpl.use('Agg')

from keras.engine.topology import Layer
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, Dense, Flatten

from keras.optimizers import Adam

from utils.MotionCorrection.network_block import encode, encode_shared, decode
from utils.MotionCorrection.customLoss import *
from utils.Unpatching import *
from utils.MotionCorrection.plot import *


class CustomLossLayer(Layer):
    def __init__(self, dHyper, patchSize, **kwargs):
        self.dHyper = dHyper
        self.patchSize = patchSize
        super(CustomLossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x_ref = inputs[0]
        decoded_ref2ref = inputs[1]
        decoded_art2ref = inputs[2]
        z_log_var = inputs[3]
        z_mean = inputs[4]

        # compute KL loss
        loss_kl = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        self.add_loss(self.dHyper['kl_weight']*K.mean(loss_kl))

        # compute MSE loss
        mse_loss_ref2ref, mse_loss_art2ref = compute_mse_loss(self.dHyper, x_ref, decoded_ref2ref, decoded_art2ref)
        self.add_loss(self.dHyper['mse_weight'] * (self.dHyper['loss_ref2ref']*mse_loss_ref2ref + self.dHyper['loss_art2ref']*mse_loss_art2ref))

        # compute gradient entropy
        ge_ref2ref, ge_art2ref = compute_gradient_entropy(self.dHyper, decoded_ref2ref, decoded_art2ref, self.patchSize)
        self.add_loss(self.dHyper['ge_weight'] * (self.dHyper['loss_ref2ref']*ge_ref2ref + self.dHyper['loss_art2ref']*ge_art2ref))

        # compute TV loss
        tv_ref2ref, tv_art2ref = compute_tv_loss(self.dHyper, decoded_ref2ref, decoded_art2ref, self.patchSize)
        self.add_loss(self.dHyper['tv_weight'] * (self.dHyper['loss_ref2ref']*tv_ref2ref + self.dHyper['loss_art2ref']*tv_art2ref))

        # compute perceptual loss
        perceptual_loss_ref2ref, perceptual_loss_art2ref = compute_perceptual_loss(x_ref, decoded_ref2ref, decoded_art2ref, self.patchSize, self.dHyper['pl_network'],self.dHyper['loss_model'])
        self.add_loss(self.dHyper['perceptual_weight'] * (self.dHyper['loss_ref2ref'] * perceptual_loss_ref2ref + self.dHyper['loss_art2ref'] * perceptual_loss_art2ref))

        return [decoded_ref2ref, decoded_art2ref]


def build_vae(patchSize, dHyper):
    # input corrupted and non-corrupted image
    x_ref = Input(shape=(1, patchSize[0], patchSize[1]))
    x_art = Input(shape=(1, patchSize[0], patchSize[1]))

    # create respective encoders
    encoded_ref = encode(x_ref, patchSize)
    encoded_art = encode(x_art, patchSize)

    # concatenate the encoded features together
    combined = concatenate([encoded_ref, encoded_art], axis=0)

    # create the shared encoder
    z, z_mean, z_log_var = encode_shared(combined, patchSize)

    # create the decoder
    decoded = decode(z, patchSize, dHyper['dropout'])

    # separate the concatenated images
    decoded_ref2ref = Lambda(lambda input: input[:input.shape[0]//2, :, :, :], output_shape=(1, patchSize[0], patchSize[1]))(decoded)
    decoded_art2ref = Lambda(lambda input: input[input.shape[0]//2:, :, :, :], output_shape=(1, patchSize[0], patchSize[1]))(decoded)

    # input to CustomLoss Layer
    [decoded_ref2ref, decoded_art2ref] = CustomLossLayer(dHyper, patchSize)([x_ref, decoded_ref2ref, decoded_art2ref, z_log_var, z_mean])

    # generate the VAE and encoder model
    vae = Model([x_ref, x_art], [decoded_ref2ref, decoded_art2ref])

    return vae


def build_discriminator(patchSize):
    def d_block(layer_input, filters, strides=1, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8, axis=1)(d)
        return d

    # define number of filters
    df = 32

    # Input image
    d0 = Input(shape=(1, patchSize[0], patchSize[1]))

    d1 = d_block(d0, df, bn=False)
    d2 = d_block(d1, df, strides=2)
    d3 = d_block(d2, df*2)
    d4 = d_block(d3, df*2, strides=2)
    d5 = d_block(d4, df*4)
    d6 = d_block(d5, df*4, strides=2)
    flat = Flatten()(d6)
    d7 = Dense(df*8)(flat)
    d8 = LeakyReLU(alpha=0.2)(d7)
    validity = Dense(1, activation='sigmoid')(d8)

    return Model(d0, validity)


def fTrain(dData, dParam, dHyper):
    # parse inputs
    batchSize = [128] if dParam['batchSize'] is None else dParam['batchSize']
    learningRate = [0.001] if dParam['learningRate'] is None else dParam['learningRate']
    epochs = 300 if dParam['epochs'] is None else dParam['epochs']

    for iBatch in batchSize:
        for iLearn in learningRate:
            fTrainInner(dData, dParam['sOutPath'], dParam['patchSize'], epochs, iBatch, iLearn, dHyper)


def fTrainInner(dData, sOutPath, patchSize, epochs, batchSize, lr, dHyper):
    TRAINING_RATIO = 1

    train_ref = dData['train_ref']
    train_art = dData['train_art']
    test_ref = dData['test_ref']
    test_art = dData['test_art']

    train_ref = np.expand_dims(train_ref, axis=1)
    train_art = np.expand_dims(train_art, axis=1)
    test_ref = np.expand_dims(test_ref, axis=1)
    test_art = np.expand_dims(test_art, axis=1)

    # optimizor
    optimizer_D = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    optimizer_G = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    # create and compile VAE model
    vae = build_vae(patchSize, dHyper)
    vae.compile(optimizer=optimizer_G, loss=None)

    weights_file = sOutPath + os.sep + 'vae_weight_ps_{}_bs_{}_lr_{}_{}_BN.h5'.format(patchSize[0], batchSize, lr, dHyper['test_patient'])
    vae.load_weights(weights_file)

    # create and compile discriminator model
    D = build_discriminator(patchSize)
    D.compile(loss='binary_crossentropy', optimizer=optimizer_D, metrics=['accuracy'])

    # create and compile combined model
    fake_ref, fake_art = vae.output
    D.trainable = False
    validity_ref = D(fake_ref)
    validity_art = D(fake_art)
    combined = Model(vae.inputs, [validity_ref, validity_art])
    combined.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                     loss_weights=[200, 300],
                     optimizer=optimizer_G)

    for epoch in range(epochs):
        print("Epoch: %d" % epoch)
        print("Number of batches: %d" % int(train_ref.shape[0] // batchSize))
        minibatches_size = batchSize * TRAINING_RATIO

        for i in range(int(train_ref.shape[0] // minibatches_size)):
            train_ref_minibatches = train_ref[i * minibatches_size:(i + 1) * minibatches_size]
            train_art_minibatches = train_art[i * minibatches_size:(i + 1) * minibatches_size]
            for j in range(TRAINING_RATIO):
                ref_batch = train_ref_minibatches[j * batchSize:(j + 1) * batchSize]
                art_batch = train_art_minibatches[j * batchSize:(j + 1) * batchSize]

                fake_ref, fake_art = vae.predict([ref_batch, art_batch])
                valid = np.ones(shape=(ref_batch.shape[0], 1))
                fake = np.zeros(shape=(ref_batch.shape[0], 1))

                d_loss_real = D.train_on_batch(ref_batch, valid)
                d_loss_fake_ref = D.train_on_batch(fake_ref, fake)
                d_loss_fake_art = D.train_on_batch(fake_art, fake)

                # --------------------------------
                #  Train Discriminator every batch
                # --------------------------------
                print('==========================')
                print('Training Discriminator')
                print('loss real: ' + str(d_loss_real[0]) + '\t\taccuracy: ' + str(d_loss_real[1]))
                print('loss ref2ref: ' + str(d_loss_fake_ref[0]) + '\taccuracy: ' + str(d_loss_fake_ref[1]))
                print('loss art2ref: ' + str(d_loss_fake_art[0]) + '\taccuracy: ' + str(d_loss_fake_art[1]))
                print('==========================')

            # --------------------------------
            #  Train Generator every 5 batches
            # --------------------------------
            g_loss = combined.train_on_batch([ref_batch, art_batch], [valid, valid])
            print('==========================')
            print('Training Generator')
            print('loss generator: ' + str(g_loss[0]))
            print('loss ref2ref: ' + str(g_loss[1]))
            print('loss art2ref: ' + str(g_loss[2]))
            print('==========================')

            # progress bar
            print ("progress: %d/%d" % (i * minibatches_size, train_ref.shape[0]))


        # -------------------------
        #  save images every epoch
        # -------------------------
        print('==========================')
        print('saving testing images...')
        print('==========================')
        curDir = sOutPath + os.sep + 'result' + os.sep + str(epoch)
        os.mkdir(curDir)
        predict_ref, predict_art = vae.predict([test_ref, test_art], batchSize, verbose=1)
        test_ref_sample = np.squeeze(test_ref, axis=1)
        test_art_sample = np.squeeze(test_art, axis=1)
        predict_art = np.squeeze(predict_art, axis=1)
        test_ref_sample = fRigidUnpatchingCorrection2D(dHyper['actualSize'], test_ref_sample, 0.8)
        test_art_sample = fRigidUnpatchingCorrection2D(dHyper['actualSize'], test_art_sample, 0.8)
        predict_art = fRigidUnpatchingCorrection2D(dHyper['actualSize'], predict_art, 0.8, 'average')
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), sharex=True, sharey=True)
        ax = axes.ravel()
        plt.gray()
        label = 'MSE: {:.2f}, SSIM: {:.2f}'
        for i in range(test_ref_sample.shape[0]):
            ax[0].imshow(test_ref_sample[i])
            ax[0].set_xlabel(label.format(mean_squared_error(test_ref_sample[i], test_ref_sample[i]),
                                          ssim(test_ref_sample[i], test_ref_sample[i],
                                               data_range=(test_ref_sample[i].max() - test_ref_sample[i].min()))))
            ax[0].set_title('reference image')

            ax[1].imshow(test_art_sample[i])
            ax[1].set_xlabel(label.format(mean_squared_error(test_ref_sample[i], test_art_sample[i]),
                                          ssim(test_ref_sample[i], test_art_sample[i],
                                               data_range=(test_art_sample[i].max() - test_art_sample[i].min()))))
            ax[1].set_title('motion-affected image')

            ax[2].imshow(predict_art[i])
            ax[2].set_xlabel(label.format(mean_squared_error(test_ref_sample[i], predict_art[i]),
                                          ssim(test_ref_sample[i], predict_art[i],
                                               data_range=(predict_art[i].max() - predict_art[i].min()))))
            ax[2].set_title('corrected image')

            plt.savefig(curDir + os.sep + str(i) + '.png')
