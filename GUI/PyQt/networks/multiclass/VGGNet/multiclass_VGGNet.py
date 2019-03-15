import os.path

import keras
import numpy as np  # for algebraic operations, matrices
# from theano import functionfrom keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Flatten  # , Layer, Flatten
from keras.models import Sequential
import scipy.io as sio


# from keras.layers import containers
# from keras.layers.convolutional import ZeroPadding2D as zero2d
from tensorflow import confusion_matrix


def fTrain(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSizes=None, learningRates=None, iEpochs=None):
    # grid search on batch_sizes and learning rates
    # parse inputs
    batchSizes = 64 if batchSizes is None else batchSizes
    learningRates = 0.01 if learningRates is None else learningRates
    iEpochs = 300 if iEpochs is None else iEpochs
    for iBatch in batchSizes:
        for iLearn in learningRates:
            fTrainInner(X_train, y_train, X_test, y_test, sOutPath, patchSize, iBatch, iLearn, iEpochs)

def fTrainInner(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSize=None, learningRate=None,iEpochs=None):
    # parse inputs
    batchSize = 64 if batchSize is None else batchSize
    learningRate = 0.01 if learningRate is None else learningRate
    iEpochs = 300 if iEpochs is None else iEpochs

    print('Training(pre) CNN (VGGNet)')
    print('with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize))

    # build model
    base = VGG16(include_top=False, weights=None, input_shape=(1, 180, 180))

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base.output_shape[1:]))
    top_model.add(Dense(11, activation='softmax'))
    # top_model.load_weights('fc_model.h5')
    model = base.add(top_model)

	_, sPath = os.path.splitdrive(sOutPath)
    sPath, sFilename = os.path.split(sPath)
    sFilename, sExt = os.path.splitext(sFilename)
    model_name = sPath + '/' + sFilename + str(patchSize[0, 0]) + str(patchSize[0, 1]) + '_lr_' + str(i) + '_bs_' + str(j)
    weight_name = model_name + '_weights.h5'
    model_json = model_name + '_json'
    model_all = model_name + '_model.h5'
    model_mat = model_name + '.mat'

    if os.path.isfile(model_mat):
		return
		# no training if output file exists

	opti = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]

	model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])

	result = model.fit(X_train,
	                   y_train,
	                   validation_data=[X_test, y_test],
	                   epochs=iEpochs,
	                   batch_size=j,
	                   callbacks=callbacks,
	                   verbose=1)

	score_test, acc_test = model.evaluate(X_test, y_test, batch_size=j)

	prob_test = model.predict(X_test, j, 0)
	y_pred=np.argmax(prob_test,axis=1)
	y_test=np.argmax(y_test,axis=1)
	confusion_mat=confusion_matrix(y_test,y_pred)

	# save model
	json_string = model.to_json()
	open(model_json, 'w').write(json_string)
	# wei = cnn.get_weights()
	model.save_weights(weight_name, overwrite=True)
	# cnn.save(model_all) # keras > v0.7

	# matlab
	acc = result.history['acc']
	loss = result.history['loss']
	val_acc = result.history['val_acc']
	val_loss = result.history['val_loss']

	print('Saving results: ' + model_name)
	sio.savemat(model_name, {'model_settings': model_json,
	                         'model': model_all,
	                         'weights': weight_name,
	                         'acc': acc,
	                         'loss': loss,
	                         'val_acc': val_acc,
	                         'val_loss': val_loss,
	                         'score_test': score_test,
	                         'acc_test': acc_test,
	                         'prob_test': prob_test,
	                         'confusion_mat':confusion_mat}
				)

###############################################################################
## OPTIMIZATIONS ##
###############################################################################
def fStatistic(confusion_mat): #each column represents a predicted label, each row represents a truth label
	cm = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
	cm = np.round(cm, decimals=3)
	dim = cm.shape[0]
	BER = np.sum(np.diag(np.identity(dim) - cm), axis=0) / dim
	#Recall = np.sum(np.diag(confusion_mat)/np.sum(confusion_mat,axis=1),axis=1) / col
	#Precision = np.sum(np.diag(normal_cm),axis=1) / col
	#F1 = 2 * (Precision * Recall) / (Precision + Recall)
	return BER