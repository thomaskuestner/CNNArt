import os
import numpy as np
from skimage.measure import compare_ssim as ssim
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from keras.layers import Input, Lambda, concatenate, Dense, Reshape, Flatten, Conv3DTranspose
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from utils.MotionCorrection.network import fCreateEncoder3D_Block, fCreateDownConv3D_Block, fCreateConv3DTranspose
from utils.MotionCorrection.PerceptualLoss import addPerceptualLoss
from utils.Unpatching import fRigidUnpatchingCorrection

def sliceRef(input):
    return input[:input.shape[0]//2, :, :, :, :]

def sliceArt(input):
    return input[input.shape[0]//2:, :, :, :, :]

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 512), mean=0.,
                              stddev=1.0)
    return z_mean + K.exp(z_log_var) * epsilon

def encode(input):
    res_1 = fCreateEncoder3D_Block(filters=32)(input)
    down_1 = fCreateDownConv3D_Block(filters=64, strides=(2, 2, 2), kernel_size=(2, 2, 2))(res_1)
    res_2 = fCreateEncoder3D_Block(filters=64)(down_1)
    down_2 = fCreateDownConv3D_Block(filters=128, strides=(2, 2, 1), kernel_size=(2, 2, 1))(res_2)
    return down_2

def encode_shared(input):
    res_1 = fCreateEncoder3D_Block(filters=128)(input)
    down_1 = fCreateDownConv3D_Block(filters=256, strides=(2, 2, 1), kernel_size=(2, 2, 1))(res_1)
    flat = Flatten()(down_1)

    z_mean = Dense(512)(flat)
    z_log_var = Dense(512)(flat)

    z = Lambda(sampling, output_shape=(512,))([z_mean, z_log_var])

    return z, z_mean, z_log_var

def decode(input):
    dense = Dense(25600*5)(input)
    reshape = Reshape((256, 10, 10, 5))(dense)
    output = fCreateConv3DTranspose(filters=256, strides=(2, 2, 1))(reshape)
    output = fCreateConv3DTranspose(filters=256, strides=(1, 1, 1))(output)
    output = fCreateConv3DTranspose(filters=128, strides=(2, 2, 1))(output)
    output = fCreateConv3DTranspose(filters=128, strides=(1, 1, 1))(output)
    output = fCreateConv3DTranspose(filters=64, strides=(2, 2, 2))(output)
    output = Conv3DTranspose(filters=1, kernel_size=1, strides=1, padding='same', activation='tanh')(output)
    return output

def createModel(patchSize, dHyper):
    # input corrupted and non-corrupted image
    x_ref = Input(shape=(1, patchSize[0], patchSize[1], patchSize[2]))
    x_art = Input(shape=(1, patchSize[0], patchSize[1], patchSize[2]))

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
    decoded_ref2ref = Lambda(sliceRef, output_shape=(1, patchSize[0], patchSize[1], patchSize[2]))(decoded)
    decoded_art2ref = Lambda(sliceArt, output_shape=(1, patchSize[0], patchSize[1], patchSize[2]))(decoded)

    # generate the VAE and encoder model
    vae = Model([x_ref, x_art], [decoded_ref2ref, decoded_art2ref])

    # compute kl loss
    loss_kl = - 0.5 * K.sum(1 + z_mean - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae.add_loss(dHyper['kl_weight'] * K.mean(loss_kl))

    # compute pixel to pixel loss
    loss_ref2ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3, 4])))\
                       ([Lambda(lambda x : dHyper['nScale']*x)(x_ref), Lambda(lambda x : dHyper['nScale']*x)(decoded_ref2ref)]) + 1e-6
    loss_art2ref = Lambda(lambda x: K.mean(K.sum(K.square(x[0] - x[1]), [1, 2, 3, 4])))\
                       ([Lambda(lambda x : dHyper['nScale']*x)(x_ref), Lambda(lambda x : dHyper['nScale']*x)(decoded_art2ref)]) + 1e-6

    vae.add_loss(dHyper['pixel_weight'] * (dHyper['loss_ref2ref']*loss_ref2ref + dHyper['loss_art2ref']*loss_art2ref))

    # add perceptual loss
    perceptual_loss_ref2ref, perceptual_loss_art2ref = 0, 0
    for i in range(patchSize[2]):
        tmp_x_ref = Lambda(lambda input: input[:, :, :, :, i], output_shape=(1, patchSize[0], patchSize[1]))(x_ref)
        tmp_decoded_ref2ref = Lambda(lambda input: input[:, :, :, :, i], output_shape=(1, patchSize[0], patchSize[1]))(decoded_ref2ref)
        tmp_decoded_art2ref = Lambda(lambda input: input[:, :, :, :, i], output_shape=(1, patchSize[0], patchSize[1]))(decoded_art2ref)

        tmp_perceptual_loss_ref2ref, tmp_perceptual_loss_art2ref = addPerceptualLoss(tmp_x_ref, tmp_decoded_ref2ref, tmp_decoded_art2ref, patchSize, dHyper['pl_network'], dHyper['loss_model'])
        perceptual_loss_ref2ref += tmp_perceptual_loss_ref2ref
        perceptual_loss_art2ref += tmp_perceptual_loss_art2ref

    perceptual_loss_ref2ref = perceptual_loss_ref2ref/patchSize[2]
    perceptual_loss_art2ref = perceptual_loss_art2ref/patchSize[2]

    vae.add_loss(dHyper['perceptual_weight'] * (dHyper['loss_ref2ref']*perceptual_loss_ref2ref + dHyper['loss_art2ref']*perceptual_loss_art2ref))

    return vae

def fTrain(dData, dParam, dHyper):
    # parse inputs
    batchSize = [128] if dParam['batchSize'] is None else dParam['batchSize']
    learningRate = [0.001] if dParam['learningRate'] is None else dParam['learningRate']
    epochs = 300 if dParam['epochs'] is None else dParam['epochs']

    for iBatch in batchSize:
        for iLearn in learningRate:
            fTrainInner(dData, dParam['sOutPath'], dParam['patchSize'], epochs, iBatch, iLearn, dHyper)

def fTrainInner(dData, sOutPath, patchSize, epochs, batchSize, lr, dHyper):
    train_ref = dData['train_ref']
    train_art = dData['train_art']
    test_ref = dData['test_ref']
    test_art = dData['test_art']

    train_ref = np.expand_dims(train_ref, axis=1)
    train_art = np.expand_dims(train_art, axis=1)
    test_ref = np.expand_dims(test_ref, axis=1)
    test_art = np.expand_dims(test_art, axis=1)

    vae = createModel(patchSize, dHyper)
    vae.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss=None)
    vae.summary()

    print('Training with epochs {} batch size {} learning rate {}'.format(epochs, batchSize, lr))

    weights_file = sOutPath + os.sep + 'vae_weight_ps_{}_bs_{}_lr_{}_{}.h5'.format(patchSize[0], batchSize, lr, dHyper['test_patient'])

    callback_list = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]
    callback_list.append(ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, period=1, save_best_only=True, save_weights_only=True))
    callback_list.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1))

    history = vae.fit([train_ref, train_art],
            shuffle=True,
            epochs=epochs,
            batch_size=batchSize,
            validation_data=([test_ref, test_art], None),
            verbose=1,
            callbacks=callback_list)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(weights_file[:-3] + '.png')

def fPredict(test_ref, test_art, dParam, dHyper):
    weights_file = dParam['sOutPath'] + os.sep + '{}.h5'.format(dHyper['bestModel'])

    patchSize = dParam['patchSize']

    vae = createModel(patchSize, dHyper)
    vae.compile(optimizer='adam', loss=None)

    vae.load_weights(weights_file)

    test_ref = np.expand_dims(test_ref, axis=1)
    test_art = np.expand_dims(test_art, axis=1)

    predict_ref, predict_art = vae.predict([test_ref, test_art], dParam['batchSize'][0], verbose=1)

    test_ref = np.squeeze(test_ref, axis=1)
    test_art = np.squeeze(test_art, axis=1)
    predict_art = np.squeeze(predict_art, axis=1)

    if dHyper['unpatch']:
        test_ref = fRigidUnpatchingCorrection(dHyper['actualSize'], test_ref, dParam['patchOverlap'])
        test_art = fRigidUnpatchingCorrection(dHyper['actualSize'], test_art, dParam['patchOverlap'])
        predict_art = fRigidUnpatchingCorrection(dHyper['actualSize'], predict_art, dParam['patchOverlap'])
        if dHyper['evaluate']:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), sharex=True, sharey=True)
            ax = axes.ravel()
            plt.gray()
            label = 'MSE: {:.2f}, SSIM: {:.2f}'
            for i in range(test_ref.shape[0]):
                ax[0].imshow(test_ref[i])
                ax[0].set_xlabel(label.format(mean_squared_error(test_ref[i], test_ref[i]), ssim(test_ref[i], test_ref[i], data_range=(test_ref[i].max() - test_ref[i].min()))))
                ax[0].set_title('reference image')

                ax[1].imshow(test_art[i])
                ax[1].set_xlabel(label.format(mean_squared_error(test_ref[i], test_art[i]), ssim(test_ref[i], test_art[i], data_range=(test_art[i].max() - test_art[i].min()))))
                ax[1].set_title('motion-affected image')

                ax[2].imshow(predict_art[i])
                ax[2].set_xlabel(label.format(mean_squared_error(test_ref[i], predict_art[i]), ssim(test_ref[i], predict_art[i], data_range=(predict_art[i].max() - predict_art[i].min()))))
                ax[2].set_title('corrected image')

                if dParam['lSave']:
                    plt.savefig(dParam['sOutPath'] + os.sep + 'result' + os.sep + str(i) + '.png')
                else:
                    plt.show()

        else:
            plt.figure()
            plt.gray()
            for i in range(predict_art.shape[0]):
                plt.imshow(predict_art[i])
                if dParam['lSave']:
                    plt.savefig(dParam['sOutPath'] + os.sep + 'result' + os.sep + str(i) + '.png')
                else:
                    plt.show()
    else:
        nPatch = predict_art.shape[0]

        for i in range(nPatch//4):
            fig, axes = plt.subplots(nrows=4, ncols=2)
            plt.gray()

            cols_title = ['original_art', 'predicted_art']

            for ax, col in zip(axes[0], cols_title):
                ax.set_title(col)

            for j in range(4):
                axes[j, 0].imshow(test_art[4*i+j])
                axes[j, 1].imshow(predict_art[4*i+j])

            if dParam['lSave']:
                plt.savefig(dParam['sOutPath'] + os.sep + 'result' + os.sep + str(i) + '.png')
            else:
                plt.show()
