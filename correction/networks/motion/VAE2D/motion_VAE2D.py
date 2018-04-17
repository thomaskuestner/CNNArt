import os
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_nrmse as nrmse
from skimage.measure import compare_psnr as psnr
from skimage.restoration import denoise_tv_chambolle
from sklearn.metrics import normalized_mutual_info_score as nmi
import matplotlib as mpl
mpl.use('Agg')

from keras.preprocessing.image import ImageDataGenerator
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
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
        # mse_loss_ref2ref, mse_loss_art2ref = compute_mse_loss(self.dHyper, x_ref, decoded_ref2ref, decoded_art2ref)
        # self.add_loss(self.dHyper['mse_weight'] * (self.dHyper['loss_ref2ref']*mse_loss_ref2ref + self.dHyper['loss_art2ref']*mse_loss_art2ref))

        # compute gradient entropy
        ge_ref2ref, ge_art2ref = compute_gradient_entropy(self.dHyper, decoded_ref2ref, decoded_art2ref, self.patchSize)
        self.add_loss(self.dHyper['ge_weight'] * (self.dHyper['loss_ref2ref']*ge_ref2ref + self.dHyper['loss_art2ref']*ge_art2ref))

        # compute TV loss
        # tv_ref2ref, tv_art2ref = compute_tv_loss(self.dHyper, decoded_ref2ref, decoded_art2ref, self.patchSize)
        # self.add_loss(self.dHyper['tv_weight'] * (self.dHyper['loss_ref2ref']*tv_ref2ref + self.dHyper['loss_art2ref']*tv_art2ref))

        # compute perceptual loss
        perceptual_loss_ref2ref, perceptual_loss_art2ref = compute_perceptual_loss(x_ref, decoded_ref2ref, decoded_art2ref, self.patchSize, self.dHyper['pl_network'],self.dHyper['loss_model'])
        self.add_loss(self.dHyper['perceptual_weight'] * (self.dHyper['loss_ref2ref'] * perceptual_loss_ref2ref + self.dHyper['loss_art2ref'] * perceptual_loss_art2ref))

        return [decoded_ref2ref, decoded_art2ref]


def createModel(patchSize, dHyper):
    # input corrupted and non-corrupted image
    x_ref = Input(shape=(1, patchSize[0], patchSize[1]))
    x_art = Input(shape=(1, patchSize[0], patchSize[1]))

    # create respective encoders
    encoded_ref = encode(x_ref, patchSize, dHyper['bn'])
    encoded_art = encode(x_art, patchSize, dHyper['bn'])

    # concatenate the encoded features together
    combined = concatenate([encoded_ref, encoded_art], axis=0)

    # create the shared encoder
    z, z_mean, z_log_var = encode_shared(combined, patchSize, dHyper['bn'])

    # create the decoder
    decoded = decode(z, patchSize, dHyper['bn'])

    # separate the concatenated images
    decoded_ref2ref = Lambda(lambda input: input[:input.shape[0]//2, :, :, :], output_shape=(1, patchSize[0], patchSize[1]))(decoded)
    decoded_art2ref = Lambda(lambda input: input[input.shape[0]//2:, :, :, :], output_shape=(1, patchSize[0], patchSize[1]))(decoded)

    # input to CustomLoss Layer
    [decoded_ref2ref, decoded_art2ref] = CustomLossLayer(dHyper, patchSize)([x_ref, decoded_ref2ref, decoded_art2ref, z_log_var, z_mean])

    # generate the VAE and encoder model
    vae = Model([x_ref, x_art], [decoded_ref2ref, decoded_art2ref])

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
    # vae.compile(optimizer=RMSprop(lr=lr), loss=None)
    vae.summary()

    print('Training with epochs {} batch size {} learning rate {}'.format(epochs, batchSize, lr))

    def gen_flow_for_two_inputs(x1, x2):
        seed = np.random.randint(0, 1e3)
        genx1 = datagen.flow(x1, batch_size=batchSize, seed=seed)
        genx2 = datagen.flow(x2, batch_size=batchSize, seed=seed)
        while True:
            X1i = genx1.next()
            X2i = genx2.next()
            yield [X1i, X2i], None


    if dHyper['augmentation']:
        weights_file = sOutPath + os.sep + 'vae_weight_ps_{}_bs_{}_lr_{}_{}_GE_augmentation_100.h5'.format(patchSize[0], batchSize, lr, dHyper['test_patient'])
        # vae.load_weights(weights_file)

        lossPlot_file = weights_file[:-3] + '.png'
        plotLoss = PlotLosses(lossPlot_file)
        callback_list = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
        callback_list.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0, verbose=1))
        callback_list.append(ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, period=1, save_best_only=True, save_weights_only=True))
        callback_list.append(plotLoss)
        datagen = ImageDataGenerator(rotation_range=10, vertical_flip=True, horizontal_flip=True)
        gen_flow = gen_flow_for_two_inputs(train_ref, train_art)

        vae.fit_generator(gen_flow,
                          shuffle=True,
                          steps_per_epoch=len(train_ref)//batchSize,
                          epochs=epochs,
                          validation_data=([test_ref, test_art], None),
                          verbose=1,
                          callbacks=callback_list)
    else:
        weights_file = sOutPath + os.sep + 'vae_weight_ps_{}_bs_{}_lr_{}_{}.h5'.format(patchSize[0], batchSize, lr, dHyper['test_patient'])
        #vae.load_weights(weights_file)

        lossPlot_file = weights_file[:-3] + '.png'
        plotLoss = PlotLosses(lossPlot_file)
        callback_list = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
        callback_list.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0, verbose=1))
        callback_list.append(ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, period=1, save_best_only=True, save_weights_only=True))
        callback_list.append(plotLoss)

        # train original dataset
        vae.fit([train_ref, train_art],
                shuffle=True,
                epochs=epochs,
                batch_size=batchSize,
                validation_data=([test_ref, test_art], None),
                verbose=1,
                callbacks=callback_list)


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
        test_ref = fRigidUnpatchingCorrection2D(dHyper['actualSize'], test_ref, dParam['patchOverlap'])
        test_art = fRigidUnpatchingCorrection2D(dHyper['actualSize'], test_art, dParam['patchOverlap'])
        predict_art = fRigidUnpatchingCorrection2D(dHyper['actualSize'], predict_art, dParam['patchOverlap'], 'average')

        # post TV processing
        predict_art_tv_1 = denoise_tv_chambolle(predict_art, weight=1)
        predict_art_tv_3 = denoise_tv_chambolle(predict_art, weight=3)
        predict_art_tv_5 = denoise_tv_chambolle(predict_art, weight=5)

        # pre TV processing
        test_art_tv_1 = denoise_tv_chambolle(test_art, weight=1)
        test_art_tv_3 = denoise_tv_chambolle(test_art, weight=3)
        test_art_tv_5 = denoise_tv_chambolle(test_art, weight=5)


        if dHyper['evaluate']:
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15), sharex=True, sharey=True)
            ax = axes.ravel()
            plt.gray()
            label = 'NRMSE: {:.2f}, SSIM: {:.3f}, NMI: {:.3f}'
            for i in range(test_ref.shape[0]):
                # orignal reconstructed images
                ax[0].imshow(test_ref[i])
                ax[0].set_xlabel(label.format(nrmse(test_ref[i], test_ref[i]), ssim(test_ref[i], test_ref[i], data_range=(test_ref[i].max() - test_ref[i].min())), nmi(test_ref[i].flatten(), test_ref[i].flatten())))
                ax[0].set_title('reference image')

                ax[1].imshow(test_art[i])
                ax[1].set_xlabel(label.format(nrmse(test_ref[i], test_art[i]), ssim(test_ref[i], test_art[i], data_range=(test_art[i].max() - test_art[i].min())), nmi(test_ref[i].flatten(), test_art[i].flatten())))
                ax[1].set_title('motion-affected image')

                ax[2].imshow(predict_art[i])
                ax[2].set_xlabel(label.format(nrmse(test_ref[i], predict_art[i]), ssim(test_ref[i], predict_art[i], data_range=(predict_art[i].max() - predict_art[i].min())), nmi(test_ref[1].flatten(), predict_art[i].flatten())))
                ax[2].set_title('original reconstructed image')

                # post TV-processing
                ax[3].imshow(predict_art_tv_1[i])
                ax[3].set_xlabel(label.format(nrmse(test_ref[i], predict_art_tv_1[i]), ssim(test_ref[i], predict_art_tv_1[i], data_range=(predict_art_tv_1[i].max() - predict_art_tv_1[i].min())), nmi(test_ref[i].flatten(), predict_art_tv_1[i].flatten())))
                ax[3].set_title('Post TV weight 1')

                ax[4].imshow(predict_art_tv_3[i])
                ax[4].set_xlabel(label.format(nrmse(test_ref[i], predict_art_tv_3[i]), ssim(test_ref[i], predict_art_tv_3[i], data_range=(predict_art_tv_3[i].max() - predict_art_tv_3[i].min())), nmi(test_ref[i].flatten(), predict_art_tv_3[i].flatten())))
                ax[4].set_title('Post TV weight 3')

                ax[5].imshow(predict_art_tv_5[i])
                ax[5].set_xlabel(label.format(nrmse(test_ref[i], predict_art_tv_5[i]), ssim(test_ref[i], predict_art_tv_5[i], data_range=(predict_art_tv_5[i].max() - predict_art_tv_5[i].min())), nmi(test_ref[i].flatten(), predict_art_tv_5[i].flatten())))
                ax[5].set_title('Post TV weight 5')

                # pre TV processing
                ax[6].imshow(test_art_tv_1[i])
                ax[6].set_xlabel(label.format(nrmse(test_ref[i], test_art_tv_1[i]), ssim(test_ref[i], test_art_tv_1[i], data_range=(test_art_tv_1[i].max() - test_art_tv_1[i].min())), nmi(test_ref[i].flatten(), test_art_tv_1[i].flatten())))
                ax[6].set_title('pre TV weight 1')

                ax[7].imshow(test_art_tv_3[i])
                ax[7].set_xlabel(label.format(nrmse(test_ref[i], test_art_tv_3[i]), ssim(test_ref[i], test_art_tv_3[i], data_range=(test_art_tv_3[i].max() - test_art_tv_3[i].min())), nmi(test_ref[i].flatten(), test_art_tv_3[i].flatten())))
                ax[7].set_title('pre TV weight 3')

                ax[8].imshow(test_art_tv_5[i])
                ax[8].set_xlabel(label.format(nrmse(test_ref[i], test_art_tv_5[i]), ssim(test_ref[i], test_art_tv_5[i], data_range=(test_art_tv_5[i].max() - test_art_tv_5[i].min())), nmi(test_ref[i].flatten(), test_art_tv_5[i].flatten())))
                ax[8].set_title('pre TV weight 5')

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
