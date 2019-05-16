import os
import os.path
import keras
import keras.backend as K
from keras.models import model_from_json
import tensorflow as tf

from GUI.PyQt.utils.label import *

from sklearn.metrics import classification_report, confusion_matrix
from GUI.PyQt.DLart.Constants_DLart import *


def predict_model(X_test, Y_test, sModelPath, batch_size=32, dlart_handle=None):
    X_test = np.expand_dims(X_test, axis=-1)

    # pathes
    _, sPath = os.path.splitdrive(sModelPath)
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)

    # load weights and model
    with open(sModelPath + os.sep + sFilename + '.json', 'r') as fp:
        model_string = fp.read()

    model = model_from_json(model_string)

    # create optimizer

    if dlart_handle is not None:
        if dlart_handle.getOptimizer() == SGD_OPTIMIZER:
            opti = keras.optimizers.SGD(momentum=dlart_handle.getMomentum(),
                                        decay=dlart_handle.getWeightDecay(),
                                        nesterov=dlart_handle.getNesterovEnabled())

        elif dlart_handle.getOptimizer() == RMS_PROP_OPTIMIZER:
            opti = keras.optimizers.RMSprop(decay=dlart_handle.getWeightDecay())

        elif dlart_handle.getOptimizer() == ADAGRAD_OPTIMIZER:
            opti = keras.optimizers.Adagrad(epsilon=None, decay=dlart_handle.getWeightDecay())

        elif dlart_handle.getOptimizer() == ADADELTA_OPTIMIZER:
            opti = keras.optimizers.Adadelta(rho=0.95, epsilon=None,
                                             decay=dlart_handle.getWeightDecay())

        elif dlart_handle.getOptimizer() == ADAM_OPTIMIZER:
            opti = keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=None,
                                         decay=dlart_handle.getWeightDecay())
        else:
            raise ValueError("Unknown Optimizer!")
    else:
        opti = keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
    model.load_weights(sModelPath + os.sep + sFilename + '_weights.h5')

    # evaluate model on test data
    score_test, acc_test = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('loss' + str(score_test) + '   acc:' + str(acc_test))

    # predict test dataset
    probability_predictions = model.predict(X_test, batch_size=batch_size, verbose=1, steps=None)

    # classification report
    # target_names = []
    # if len(classMappings[list(classMappings.keys())[0]]) == 3:
    #     for i in sorted(classMappings):
    #         i = i % 100
    #         i = i % 10
    #         if Label.LABEL_STRINGS[i] not in target_names:
    #             target_names.append(Label.LABEL_STRINGS[i])
    # elif len(classMappings[list(classMappings.keys())[0]]) == 8:
    #     for i in sorted(classMappings):
    #         i = i % 100
    #         if Label.LABEL_STRINGS[i] not in target_names:
    #             target_names.append(Label.LABEL_STRINGS[i])
    # else:
    #     for i in sorted(classMappings):
    #         target_names.append(Label.LABEL_STRINGS[i])

    classification_summary = classification_report(np.argmax(Y_test, axis=1),
                                                   np.argmax(probability_predictions, axis=1),
                                                   target_names=None, digits=4)

    # confusion matrix
    confusionMatrix = confusion_matrix(y_true=np.argmax(Y_test, axis=1),
                                       y_pred=np.argmax(probability_predictions, axis=1),
                                       labels=range(int(probability_predictions.shape[1])))

    prediction = {
        'predictions': probability_predictions,
        'score_test': score_test,
        'acc_test': acc_test,
        'classification_report': classification_summary,
        'confusion_matrix': confusionMatrix
    }

    return prediction


def predict_segmentation_model(X_test, y_test=None, Y_segMasks_test=None, sModelPath=None, sOutPath=None, batch_size=64,
                               usingClassification=False, dlart_handle=None):
    """Takes an already trained model and computes the loss and Accuracy over the samples X with their Labels y
        Input:
            X: Samples to predict on. The shape of X should fit to the input shape of the model
            y: Labels for the Samples. Number of Samples should be equal to the number of samples in X
            sModelPath: (String) full path to a trained keras model. It should be *_json.txt file. there has to be a corresponding *_weights.h5 file in the same directory!
            sOutPath: (String) full path for the Output. It is a *.mat file with the computed loss and accuracy stored.
                        The Output file has the Path 'sOutPath'+ the filename of sModelPath without the '_json.txt' added the suffix '_pred.mat'
            batchSize: Batchsize, number of samples that are processed at once"""

    X_test = np.expand_dims(X_test, axis=-1)
    Y_segMasks_test_foreground = np.expand_dims(Y_segMasks_test, axis=-1)
    Y_segMasks_test_background = np.ones(Y_segMasks_test_foreground.shape) - Y_segMasks_test_foreground
    Y_segMasks_test = np.concatenate((Y_segMasks_test_background, Y_segMasks_test_foreground), axis=-1)

    # if usingClassification:
    #   y_test = np.expand_dims(y_test, axis=-1)

    _, sPath = os.path.splitdrive(sModelPath)
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)

    listdir = os.listdir(sModelPath)

    # sModelPath = sModelPath.replace("_json.txt", "")
    # weight_name = sModelPath + '_weights.h5'
    # model_json = sModelPath + '_json.txt'
    # model_all = sModelPath + '_model.h5'

    # load weights and model (new way)
    with open(sModelPath + os.sep + sFilename + '.json', 'r') as fp:
        model_string = fp.read()

    model = model_from_json(model_string)

    model.summary()

    if dlart_handle is not None:
        if dlart_handle.getOptimizer() == SGD_OPTIMIZER:
            opti = keras.optimizers.SGD(momentum=dlart_handle.getMomentum(),
                                        decay=dlart_handle.getWeightDecay(),
                                        nesterov=dlart_handle.getNesterovEnabled())

        elif dlart_handle.getOptimizer() == RMS_PROP_OPTIMIZER:
            opti = keras.optimizers.RMSprop(decay=dlart_handle.getWeightDecay())

        elif dlart_handle.getOptimizer() == ADAGRAD_OPTIMIZER:
            opti = keras.optimizers.Adagrad(epsilon=None, decay=dlart_handle.getWeightDecay())

        elif dlart_handle.getOptimizer() == ADADELTA_OPTIMIZER:
            opti = keras.optimizers.Adadelta(rho=0.95, epsilon=None,
                                             decay=dlart_handle.getWeightDecay())

        elif dlart_handle.getOptimizer() == ADAM_OPTIMIZER:
            opti = keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=None,
                                         decay=dlart_handle.getWeightDecay())
        else:
            raise ValueError("Unknown Optimizer!")
    else:
        opti = keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    if usingClassification:
        model.compile(loss={'segmentation_output': dice_coef_loss, 'classification_output': 'categorical_crossentropy'},
                      optimizer=opti,
                      metrics={'segmentation_output': dice_coef, 'classification_output': 'accuracy'})

        model.load_weights(sModelPath + os.sep + sFilename + '_weights.h5')

        loss_test, segmentation_output_loss_test, classification_output_loss_test, segmentation_output_dice_coef_test, classification_output_acc_test \
            = model.evaluate(X_test,
                             {'segmentation_output': Y_segMasks_test, 'classification_output': y_test},
                             batch_size=batch_size, verbose=1)

        print('loss' + str(loss_test) + ' segmentation loss:' + str(
            segmentation_output_loss_test) + ' classification loss: ' + str(classification_output_loss_test) + \
              ' segmentation dice coef: ' + str(
            segmentation_output_dice_coef_test) + ' classification accuracy: ' + str(classification_output_acc_test))

        prob_pre = model.predict(X_test, batch_size=batch_size, verbose=1)

        predictions = {'prob_pre': prob_pre,
                       'loss_test': loss_test,
                       'segmentation_output_loss_test': segmentation_output_loss_test,
                       'classification_output_loss_test': classification_output_loss_test,
                       'segmentation_output_dice_coef_test': segmentation_output_dice_coef_test,
                       'classification_output_acc_test': classification_output_acc_test}
    else:
        model.compile(loss=dice_coef_loss, optimizer=opti, metrics=[dice_coef])
        model.load_weights(sModelPath + os.sep + sFilename + '_weights.h5')

        score_test, acc_test = model.evaluate(X_test, Y_segMasks_test, batch_size=batch_size)
        print('loss: ' + str(score_test) + '   dice coef:' + str(acc_test))

        prob_pre = model.predict(X_test, batch_size=batch_size, verbose=1)

        predictions = {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test}

    return predictions


def dice_coef(y_true, y_pred, epsilon=1e-5):
    dice_numerator = 2.0 * K.sum(y_true * y_pred, axis=[1, 2, 3, 4])
    dice_denominator = K.sum(K.square(y_true), axis=[1, 2, 3, 4]) + K.sum(K.square(y_pred), axis=[1, 2, 3, 4])

    dice_score = dice_numerator / (dice_denominator + epsilon)
    return K.mean(dice_score, axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_2(ground_truth, prediction, weight_map=None):
    """
    Function to calculate the dice loss with the definition given in

        Milletari, F., Navab, N., & Ahmadi, S. A. (2016)
        V-net: Fully convolutional neural
        networks for volumetric medical image segmentation. 3DV 2016

    using a square in the denominator

    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    ground_truth = tf.to_int64(ground_truth)
    prediction = tf.cast(prediction, tf.float32)
    ids = tf.range(tf.to_int64(tf.shape(ground_truth)[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(tf.shape(prediction)))
    if weight_map is not None:
        n_classes = prediction.shape[1].value
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(weight_map_nclasses * tf.square(prediction),
                          reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot * weight_map_nclasses,
                                 reduction_axes=[0])
    else:
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(tf.square(prediction), reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
    epsilon_denominator = 0.00001

    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    # dice_score.set_shape([n_classes])
    # minimising (1 - dice_coefficients)

    # return 1.0 - tf.reduce_mean(dice_score)
    return tf.reduce_mean(dice_score)
