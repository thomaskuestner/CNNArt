import os.path
import scipy.io as sio
import numpy as np  # for algebraic operations, matrices
import keras.models
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout  # , Layer, Flatten
# from keras.layers import containers
from keras.models import model_from_json,Model,load_model
from keras_applications.densenet import DenseNet
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK
from sklearn.metrics import confusion_matrix
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D as pool2
from keras.callbacks import EarlyStopping,ModelCheckpoint
# from keras.layers.convolutional import ZeroPadding2D as zero2d
from keras.regularizers import l2  # , activity_l2
# from theano import functionfrom keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD
from keras.layers.merge import concatenate
from keras.layers import Input,add
from keras.layers.advanced_activations import PReLU,ELU
from keras.layers.pooling import GlobalAveragePooling2D
import h5py


def fTrain(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSizes=None, learningRates=None, iEpochs=None):
# grid search on batch_sizes and learning rates
# parse inputs
	batchSizes = 64 if batchSizes is None else batchSizes
	learningRates = 0.01 if learningRates is None else learningRates
	iEpochs = 300 if iEpochs is None else iEpochs
	for iBatch in batchSizes:
		for iLearn in learningRates:
			fTrainInner(X_train, y_train, X_test, y_test, sOutPath, patchSize, iBatch, iLearn, iEpochs)

def fTrainInner(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSize=None, learningRate=None, iEpochs=None):
	# parse inputs
	batchSize = 64 if batchSize is None else batchSize
	learningRate = 0.01 if learningRate is None else learningRate
	iEpochs = 300 if iEpochs is None else iEpochs

	print('Training DenseNet')
	print('with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize))

	# save names
	_, sPath = os.path.splitdrive(sOutPath)
	sPath, sFilename = os.path.split(sPath)
	sFilename, sExt = os.path.splitext(sFilename)
	model_name = sPath + '/' + sFilename + str(patchSize[0, 0]) + str(patchSize[0, 1]) + '_lr_' + str(
		learningRate) + '_bs_' + str(batchSize)
	weight_name = model_name + '_weights.h5'
	model_json = model_name + '_json'
	model_all = model_name + '_model.h5'
	model_mat = model_name + '.mat'

	if (os.path.isfile(model_mat)):  # no training if output file exists
		return

	# create model
	cnn = DenseNet(nb_classes=11, img_dim=(1, patchSize[0, 0], patchSize[0, 1]), nb_dense_block=2, depth=19, growth_rate=16,
				   nb_filter=64)

	# opti = SGD(lr=learningRate, momentum=1e-8, decay=0.1, nesterov=True);#Adag(lr=0.01, epsilon=1e-06)
	opti = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	callbacks = [
		ModelCheckpoint(filepath=model_name + 'bestweights.hdf5', monitor='val_loss', verbose=0, save_best_only=True,
						save_weights_only=False)]

	cnn.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
	cnn.summary()
	result = cnn.fit(X_train,
					 y_train,
					 validation_data=[X_test, y_test],
					 epochs=iEpochs,
					 batch_size=batchSize,
					 callbacks=callbacks,
					 verbose=1)

	score_test, acc_test = cnn.evaluate(X_test, y_test, batch_size=batchSize)

	prob_test = cnn.predict(X_test, batchSize, 0)
	y_pred = np.argmax(prob_test, axis=1)
	y_test = np.argmax(y_test, axis=1)
	confusion_mat = confusion_matrix(y_test, y_pred)

	# save model
	json_string = cnn.to_json()
	open(model_json, 'w').write(json_string)
	# wei = cnn.get_weights()
	cnn.save_weights(weight_name, overwrite=True)
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
							 'confusion_mat': confusion_mat})


def fPredict(X_test, y_test, model_name, sOutPath, patchSize, batchSize):
	weight_name = model_name[0]
	#model_json = model_name[1] + '_json'
	#model_all = model_name[0] + '.hdf5'
	_, sPath = os.path.splitdrive(sOutPath)
	sPath, sFilename = os.path.split(sOutPath)
	#sFilename, sExt = os.path.splitext(sFilename)

	#f = h5py.File(weight_name, 'r+')
	#del f['optimizer_weights']
	#f.close()
	model=load_model(weight_name)
	opti = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1)]

	#model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
	#model.load_weights(weight_name)
	model.summary()

	score_test, acc_test = model.evaluate(X_test, y_test, batch_size=batchSize)
	prob_pre = model.predict(X_test, batchSize, 0)

	y_pred=np.argmax(prob_pre,axis=1)
	y_test=np.argmax(y_test,axis=1)
	confusion_mat=confusion_matrix(y_test,y_pred)
	# modelSave = model_name[:-5] + '_pred.mat'
	modelSave = sOutPath + '/' + sFilename + '_result.mat'
	sio.savemat(modelSave, {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test, 'confusion_mat':confusion_mat})



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

## helper functions
def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

