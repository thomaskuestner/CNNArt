import os
import numpy as np
from skimage.measure import compare_ssim as ssim
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
mpl.use('Agg')

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from utils.MotionCorrection.network_block import encode, encode_shared, decode
from utils.MotionCorrection.customLoss import *
from utils.Unpatching import *
from utils.MotionCorrection.plot import *


def createModel(patchSize, dHyper):
    # input corrupted and non-corrupted image
    x_ref = Input(shape=(1, patchSize[0], patchSize[1], patchSize[2]))
    x_art = Input(shape=(1, patchSize[0], patchSize[1], patchSize[2]))

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
    decoded_ref2ref = Lambda(lambda input: input[:input.shape[0]//2, :, :, :, :], output_shape=(1, patchSize[0], patchSize[1], patchSize[2]))(decoded)
    decoded_art2ref = Lambda(lambda input: input[input.shape[0]//2:, :, :, :, :], output_shape=(1, patchSize[0], patchSize[1], patchSize[2]))(decoded)

    # generate the VAE and encoder model
    vae = Model([x_ref, x_art], [decoded_ref2ref, decoded_art2ref])

    # compute kl loss
    loss_kl = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae.add_loss(dHyper['kl_weight'] * K.mean(loss_kl))

    # compute pixel to pixel loss
    loss_ref2ref, loss_art2ref = compute_mse_loss(dHyper, x_ref, decoded_ref2ref, decoded_art2ref)
    vae.add_loss(dHyper['mse_weight'] * (dHyper['loss_ref2ref']*loss_ref2ref + dHyper['loss_art2ref']*loss_art2ref))

    # add perceptual loss
    perceptual_loss_ref2ref, perceptual_loss_art2ref = compute_perceptual_loss(x_ref, decoded_ref2ref, decoded_art2ref, patchSize, dHyper['pl_network'], dHyper['loss_model'])
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
    lossPlot_file = weights_file[:-3] + '.png'

    plotLoss = PlotLosses(lossPlot_file)

    callback_list = []
    # callback_list = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]
    callback_list.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0, verbose=1))
    callback_list.append(ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, period=1, save_best_only=True, save_weights_only=True))
    callback_list.append(plotLoss)

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
        test_ref = fRigidUnpatchingCorrection3D(dHyper['actualSize'], test_ref, dParam['patchOverlap'])
        test_art = fRigidUnpatchingCorrection3D(dHyper['actualSize'], test_art, dParam['patchOverlap'])
        predict_art = fRigidUnpatchingCorrection3D(dHyper['actualSize'], predict_art, dParam['patchOverlap'], mode='average')
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
