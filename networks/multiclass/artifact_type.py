import os.path
import scipy.io as sio
import numpy as np  # for algebraic operations, matrices
import keras.models
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout  # , Layer, Flatten
# from keras.layers import containers
from keras.models import model_from_json,Model,load_model
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
from Densenet import DenseNet


#temp/Dense ResNet for 4040
def Block(input,num_filters,with_shortcut):
	out1 = Conv2D(filters=num_filters/2, kernel_size=(1, 1), kernel_initializer='he_normal', weights=None, padding='same',
	              strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(input)
	out2 = Conv2D(filters=num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None, padding='same',
	              strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(out1)
	out3 = Conv2D(filters=num_filters, kernel_size=(1, 1), kernel_initializer='he_normal', weights=None, padding='same',
	              strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(out2)
	# out4 = pool2(pool_size=(3, 3), strides=(2, 2), data_format="channel_first")(out3)

	if with_shortcut:
		input = Conv2D(filters=num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None,
		              padding='same',strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(input)
		return add([input,out3])
	else:
		input = Conv2D(filters=num_filters, kernel_size=(1, 1), kernel_initializer='he_normal', weights=None,
		              padding='same',strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(input)
		return add([input,out3])

def create4040Model(patchSize):
	seed=5
	np.random.seed(seed)
	input=Input(shape=(1,patchSize[0, 0], patchSize[0, 1]))
	out=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(input)
	out1=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out)
	#out1=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	out=Block(out1,128,with_shortcut=True)
	out=Block(out,128,with_shortcut=False)
	out=concatenate(inputs=[out1,out],axis=1)
	out2=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	out=Block(out2,256,with_shortcut=True)
	out=Block(out,256,with_shortcut=False)
	out1=pool2(pool_size=(3,3),strides=(2,2),data_format="channels_first")(out1)
	out=concatenate(inputs=[out1,out2,out],axis=1)
	out3=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	#out5=GlobalAveragePooling2D(data_format='channels_first')(out3)

	out5=Flatten()(out3)

	out6=Dense(units=11,
	           kernel_initializer='normal',
               kernel_regularizer='l2',
	           activation='softmax')(out5)

	cnn = Model(inputs=input,outputs=out6)
	return cnn

#temp/Inception-ResNet for 180180
def create180180Model(patchSize):
	seed=5
	np.random.seed(seed)
	input=Input(shape=(1,patchSize[0, 0], patchSize[0, 1]))
	out1=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(input)
	out2=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out1)
	out2=pool2(pool_size=(2,2),data_format='channels_first')(out2)

	out3=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out2)
	out4=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out3)
	out4=add([out2,out4])
	out4=pool2(pool_size=(2,2),data_format='channels_first')(out4)

	out_3=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out4)
	out_4=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out_3)

	out5_1=Conv2D(filters=32,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out_4)

	out5_2=Conv2D(filters=32,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out_4)
	out5_2=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5_2)

	out5_3=Conv2D(filters=32,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out_4)
	out5_3=Conv2D(filters=128,kernel_size=(5,5),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5_3)

	out5_4=pool2(pool_size=(3,3),strides=(1,1),padding='same',data_format='channels_first')(out_4)
	out5_4=Conv2D(filters=128,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5_4)

	out5=concatenate(inputs=[out5_1,out5_2,out5_3],axis=1)

	out7=Conv2D(filters=288,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),
	            activation='relu')(out5)
	out7=add([out5, out7])
	out7=pool2(pool_size=(2,2),data_format='channels_first')(out7)
	sout7=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),
	            activation='relu')(out7)

	out8=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),
	            activation='relu')(out7)
	out9=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),
	            activation='relu')(out8)
	out9=add([sout7, out9])

	out9=pool2(pool_size=(2,2),data_format='channels_first')(out9)

	out10=Flatten()(out9)


	out11=Dense(units=11,
	           kernel_initializer='normal',
               kernel_regularizer='l2',
	           activation='softmax')(out10)

	cnn = Model(inputs=input,outputs=out11)
	return cnn


def fTrain(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSize=None, learningRate=None, iEpochs=None):
	# parse inputs
	batchSize = 64 if batchSize is None else batchSize
	learningRate = 0.01 if learningRate is None else learningRate
	iEpochs = 300 if iEpochs is None else iEpochs

	print 'Training CNN'
	print 'with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize)

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
	if (patchSize[0,0]==40 & patchSize[0,1]==40):
		cnn = create4040Model(patchSize)
	else:
		if(patchSize[0,0]==180 & patchSize[0,1]==180):
			cnn = create180180Model(patchSize)
		else:
			print 'NO models for patch size ' + patchSize[0,0] + patchSize[0,0]


	# opti = SGD(lr=learningRate, momentum=1e-8, decay=0.1, nesterov=True);#Adag(lr=0.01, epsilon=1e-06)
	opti = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=1), ModelCheckpoint(filepath=model_name+'bestweights.hdf5',monitor='val_acc',verbose=0,save_best_only=True,save_weights_only=False)]
	#callbacks = [ModelCheckpoint(filepath=model_name+'bestweights.hdf5',monitor='val_acc',verbose=0,save_best_only=True,save_weights_only=False)]

	cnn.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
	cnn.summary()
	result = cnn.fit(X_train,
	                 y_train,
	                 validation_data=[X_test, y_test],
	                 epochs=iEpochs,
	                 batch_size=batchSize,
	                 callbacks=callbacks,
	                 verbose=1)

	score_test, acc_test = cnn.evaluate(X_test, y_test, batch_size=batchSize )

	prob_test = cnn.predict(X_test, batchSize, 0)
	y_pred=np.argmax(prob_test,axis=1)
	y_test=np.argmax(y_test,axis=1)
	confusion_mat=confusion_matrix(y_test,y_pred)

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

	print 'Saving results: ' + model_name
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
	                         'confusion_mat':confusion_mat})


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
	model.summary();

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
def fHyperasTrain(X_train, Y_train, X_test, Y_test, patchSize):
	# explicitly stated here instead of cnn = createModel() to allow optimization
	cnn = Sequential()

	cnn.add(Conv2D(32,  # 64
                   7,
	               7,
	               init='normal',
	               weights=None,
	               padding='valid',
	               subsample=(1, 1),
	               W_regularizer=l2(1e-6)))
	cnn.add(Activation('relu'))

	cnn.add(Conv2D(64,  # learning rate: 0.1 -> 76%
	               3,
	               3,
	               init='normal',
	               weights=None,
                   padding='valid',
	               subsample=(1, 1),
	               W_regularizer=l2(1e-6)))
	cnn.add(Activation('relu'))

	cnn.add(Conv2D(128,  # learning rate: 0.1 -> 76%
	               3,
	               3,
	               init='normal',
	               weights=None,
	               padding='valid',
	               subsample=(1, 1),
	               W_regularizer=l2(1e-6)))
	cnn.add(Activation('relu'))

	cnn.add(Flatten())

	cnn.add(Dense(input_dim=100,
	              output_dim=2,
	              init='normal',
	              # activation = 'sigmoid',
	              W_regularizer='l2'))
	cnn.add(Activation('softmax'))

	opti = SGD(lr={{choice([0.1, 0.01, 0.05, 0.005, 0.001])}}, momentum=1e-8, decay=0.1, nesterov=True)
	cnn.compile(loss='categorical_crossentropy',
	            optimizer=opti,
                metrics=['accuracy'])

	epochs = 300

	result = cnn.fit(X_train, Y_train,
	                 batch_size=128,  # {{choice([64, 128])}}
	                 epochs=epochs,
	                 show_accuracy=True,
	                 verbose=2,
	                 validation_data=(X_test, Y_test))
	score_test, acc_test = cnn.evaluate(X_test, Y_test, verbose=0)

	return {'loss': -acc_test, 'status': STATUS_OK, 'model': cnn, 'trainresult': result, 'score_test': score_test}


def fGridTrain(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSizes, learningRates, iEpochs):
	# grid search on batch_sizes and learning rates
	for j in batchSizes:
		for i in learningRates:
			fTrain(X_train, y_train, X_test, y_test, sOutPath, patchSize, j, i, iEpochs)
#	model = KerasClassifier(build_fn=createModel, epochs=iEpochs,verbose=1)
#	param_grid = dict(batch_size=batchSizes,learn_rate=learningRates)
#	grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
#	grid_result = grid.fit(X_train, y_train)
#
#	print ("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#	means = grid_result.cv_results_['mean_test_score']
#	stds = grid_result.cv_results_['std_test_score']
#	params = grid_result.cv_results_['params']
#	for mean, st, param in zip(means, stds, params):
#		print("%f (%f) with : %r" % (mean, st, param))

def fFinetune(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSizes, learningRates, iEpochs):
	base = VGG16(include_top=False,weights=None,input_shape=(1,180,180))

	top_model = Sequential()
	top_model.add(Flatten(input_shape=base.output_shape[1:]))
	top_model.add(Dense(11, activation='softmax'))
	#top_model.load_weights('fc_model.h5')
	model=base.add(top_model)

	for j in batchSizes:
		for i in learningRates:
			print 'Training(pre) CNN'
			print 'with lr = ' + str(i) + ' , batchSize = ' + str(j)

			# save names
			_, sPath = os.path.splitdrive(sOutPath)
			sPath, sFilename = os.path.split(sPath)
			sFilename, sExt = os.path.splitext(sFilename)
			model_name = sPath + '/' + sFilename + str(patchSize[0, 0]) + str(patchSize[0, 1]) + '_lr_' + str(
				i) + '_bs_' + str(j)
			weight_name = model_name + '_weights.h5'
			model_json = model_name + '_json'
			model_all = model_name + '_model.h5'
			model_mat = model_name + '.mat'

			if (os.path.isfile(model_mat)):  # no training if output file exists
				return
			opti = keras.optimizers.Adam(lr=i, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
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

			print 'Saving results: ' + model_name
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
			                         'confusion_mat':confusion_mat})


def fStatistic(confusion_mat): #each column represents a predicted label, each row represents a truth label
	cm = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
	cm = np.round(cm, decimals=3)
	dim = cm.shape[0]
	BER = np.sum(np.diag(np.identity(dim) - cm), axis=0) / dim
	#Recall = np.sum(np.diag(confusion_mat)/np.sum(confusion_mat,axis=1),axis=1) / col
	#Precision = np.sum(np.diag(normal_cm),axis=1) / col
	#F1 = 2 * (Precision * Recall) / (Precision + Recall)
	return BER


def fDensenet(X_train, y_train, X_test, y_test, sOutPath, patchSize, batchSizes, learningRates, iEpochs=None):
	# parse inputs
	for batchSize in batchSizes:
		for learningRate in learningRates:
			batchSize = 64 if batchSize is None else batchSize
			learningRate = 0.01 if learningRate is None else learningRate
			iEpochs = 300 if iEpochs is None else iEpochs

			print 'Training DenseNet'
			print 'with lr = ' + str(learningRate) + ' , batchSize = ' + str(batchSize)

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
			cnn = DenseNet(nb_classes=11,img_dim=(1,patchSize[0, 0], patchSize[0, 1]),nb_dense_block=2,depth=19,growth_rate=16,nb_filter=64)


	# opti = SGD(lr=learningRate, momentum=1e-8, decay=0.1, nesterov=True);#Adag(lr=0.01, epsilon=1e-06)
			opti = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
			callbacks = [ModelCheckpoint(filepath=model_name+'bestweights.hdf5',monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False)]

			cnn.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
			cnn.summary()
			result = cnn.fit(X_train,
	                        y_train,
	                        validation_data=[X_test, y_test],
	                        epochs=iEpochs,
	                        batch_size=batchSize,
	                        callbacks=callbacks,
	                        verbose=1)

			score_test, acc_test = cnn.evaluate(X_test, y_test, batch_size=batchSize )

			prob_test = cnn.predict(X_test, batchSize, 0)
			y_pred=np.argmax(prob_test,axis=1)
			y_test=np.argmax(y_test,axis=1)
			confusion_mat=confusion_matrix(y_test,y_pred)

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

			print 'Saving results: ' + model_name
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
	                         'confusion_mat':confusion_mat})


## helper functions
def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

