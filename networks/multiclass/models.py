import os.path
import scipy.io as sio
import numpy as np  # for algebraic operations, matrices
import keras.models
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout  # , Layer, Flatten
# from keras.layers import containers
from keras.models import model_from_json,Model
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



#googlenet7 for 120120 180180
def createNewModel(patchSize):
	seed=5
	np.random.seed(seed)
	input=Input(shape=(1,patchSize[0, 0], patchSize[0, 1]))
	out1=Conv2D(filters=32,
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
				activation='relu')(input)

	out2=Conv2D(filters=64,
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
				activation='relu')(out1)
	out2=pool2(pool_size=(2,2),data_format='channels_first')(out2)

	out3=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
				activation='relu')(out2)


	out4=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
				activation='relu')(out3)
	out4=pool2(pool_size=(2,2),data_format='channels_first')(out4)

	out5_1=Conv2D(filters=32,
				  kernel_size=(1,1),
				  kernel_initializer='he_normal',
				  weights=None,
				  padding='same',
				  strides=(1, 1),
				  kernel_regularizer=l2(1e-6),
	              activation='relu')(out4)

	out5_2=Conv2D(filters=32,  # learning rate: 0.1 -> 76%
				  kernel_size=(1,1),
				  kernel_initializer='he_normal',
				  weights=None,
				  padding='same',
				  strides=(1, 1),
				  kernel_regularizer=l2(1e-6),
	              activation='relu')(out4)
	out5_2=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
				  kernel_size=(3,3),
				  kernel_initializer='he_normal',
				  weights=None,
				  padding='same',
				  strides=(1, 1),
				  kernel_regularizer=l2(1e-6),
	              activation='relu')(out5_2)

	out5_3=Conv2D(filters=32,  # learning rate: 0.1 -> 76%
				  kernel_size=(1,1),
				  kernel_initializer='he_normal',
				  weights=None,
				  padding='same',
				  strides=(1, 1),
				  kernel_regularizer=l2(1e-6),
	              activation='relu')(out4)
	out5_3=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
				  kernel_size=(5,5),
				  kernel_initializer='he_normal',
				  weights=None,
				  padding='same',
				  strides=(1, 1),
				  kernel_regularizer=l2(1e-6),
	              activation='relu')(out5_3)

	out5_4=pool2(pool_size=(3,3),strides=(1,1),padding='same',data_format='channels_first')(out4)
	out5_4=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
				  kernel_size=(1,1),
				  kernel_initializer='he_normal',
				  weights=None,
				  padding='same',
				  strides=(1, 1),
				  kernel_regularizer=l2(1e-6),
	              activation='relu')(out5_4)

	out5=concatenate(inputs=[out5_1,out5_2,out5_3],axis=1)

	out7=Conv2D(filters=256,  # learning rate: 0.1 -> 76%
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
	            activation='relu')(out5)
	#out7=pool2(pool_size=(2,2),data_format='channels_first')(out7)

	out8=Conv2D(filters=256,  # learning rate: 0.1 -> 76%
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
	            activation='relu')(out7)
	out8=pool2(pool_size=(2,2),data_format='channels_first')(out8)

	out9=Flatten()(out8)


	out10=Dense(units=11,
	           kernel_initializer='normal',
               kernel_regularizer='l2',
	           activation='softmax')(out9)

	cnn = Model(inputs=input,outputs=out10)
	return cnn

#336C2P for 4040
def createNewModel(patchSize):
	seed=5
	np.random.seed(seed)
	input=Input(shape=(1,patchSize[0, 0], patchSize[0, 1]))
	out1=Conv2D(filters=32,
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
				activation='relu')(input)

	out2=Conv2D(filters=64,
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
				activation='relu')(out1)

	out3=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
				activation='relu')(out2)

	out4=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
				activation='relu')(out3)
	out4=pool2(pool_size=(2,2),data_format='channels_first')(out4)

	out5=Conv2D(filters=256,
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
	            activation='relu')(out4)

	out6=Conv2D(filters=256,  # learning rate: 0.1 -> 76%
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
	            activation='relu')(out5)
	out6=pool2(pool_size=(2,2),data_format='channels_first')(out6)

	out7=Flatten()(out6)

	out10=Dense(units=11,
	           kernel_initializer='normal',
               kernel_regularizer='l2',
	           activation='softmax')(out7)

	cnn = Model(inputs=input,outputs=out10)
	return cnn

#temp/googlenet764
def createNewModel(patchSize):
	seed=5
	np.random.seed(seed)
	input=Input(shape=(1,patchSize[0, 0], patchSize[0, 1]))
	out1=Conv2D(filters=64,
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
				activation='relu')(input)

	out2=Conv2D(filters=64,
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
				activation='relu')(out1)
	out2=pool2(pool_size=(2,2),data_format='channels_first')(out2)

	out3=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
				activation='relu')(out2)


	out4=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
				activation='relu')(out3)
	out4=pool2(pool_size=(2,2),data_format='channels_first')(out4)

	out5_1=Conv2D(filters=32,
				  kernel_size=(1,1),
				  kernel_initializer='he_normal',
				  weights=None,
				  padding='same',
				  strides=(1, 1),
				  kernel_regularizer=l2(1e-6),
	              activation='relu')(out4)

	out5_2=Conv2D(filters=32,  # learning rate: 0.1 -> 76%
				  kernel_size=(1,1),
				  kernel_initializer='he_normal',
				  weights=None,
				  padding='same',
				  strides=(1, 1),
				  kernel_regularizer=l2(1e-6),
	              activation='relu')(out4)
	out5_2=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
				  kernel_size=(3,3),
				  kernel_initializer='he_normal',
				  weights=None,
				  padding='same',
				  strides=(1, 1),
				  kernel_regularizer=l2(1e-6),
	              activation='relu')(out5_2)

	out5_3=Conv2D(filters=32,  # learning rate: 0.1 -> 76%
				  kernel_size=(1,1),
				  kernel_initializer='he_normal',
				  weights=None,
				  padding='same',
				  strides=(1, 1),
				  kernel_regularizer=l2(1e-6),
	              activation='relu')(out4)
	out5_3=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
				  kernel_size=(5,5),
				  kernel_initializer='he_normal',
				  weights=None,
				  padding='same',
				  strides=(1, 1),
				  kernel_regularizer=l2(1e-6),
	              activation='relu')(out5_3)

	out5_4=pool2(pool_size=(3,3),strides=(1,1),padding='same',data_format='channels_first')(out4)
	out5_4=Conv2D(filters=128,  # learning rate: 0.1 -> 76%
				  kernel_size=(1,1),
				  kernel_initializer='he_normal',
				  weights=None,
				  padding='same',
				  strides=(1, 1),
				  kernel_regularizer=l2(1e-6),
	              activation='relu')(out5_4)

	out5=concatenate(inputs=[out5_1,out5_2,out5_3],axis=1)

	out7=Conv2D(filters=256,  # learning rate: 0.1 -> 76%
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
	            activation='relu')(out5)
	#out7=pool2(pool_size=(2,2),data_format='channels_first')(out7)

	out8=Conv2D(filters=256,  # learning rate: 0.1 -> 76%
				kernel_size=(3,3),
				kernel_initializer='he_normal',
				weights=None,
				padding='valid',
				strides=(1, 1),
				kernel_regularizer=l2(1e-6),
	            activation='relu')(out7)
	out8=pool2(pool_size=(2,2),data_format='channels_first')(out8)

	out9=Flatten()(out8)


	out10=Dense(units=11,
	           kernel_initializer='normal',
               kernel_regularizer='l2',
	           activation='softmax')(out9)

	cnn = Model(inputs=input,outputs=out10)
	return cnn

#temp/resgooglenet764 wrong!!!
def createNewModel(patchSize):
	seed=5
	np.random.seed(seed)
	input=Input(shape=(1,patchSize[0, 0], patchSize[0, 1]))
	out1=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(input)
	out2=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out1)
	out2=pool2(pool_size=(2,2),data_format='channels_first')(out2)

	out3=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out2)
	out4=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out3)
	out4=add([out2,out4])
	out4=pool2(pool_size=(2,2),data_format='channels_first')(out4)

	out5_1=Conv2D(filters=32,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out4)

	out5_2=Conv2D(filters=32,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out4)
	out5_2=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5_2)

	out5_3=Conv2D(filters=32,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out4)
	out5_3=Conv2D(filters=128,kernel_size=(5,5),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5_3)

	out5_4=pool2(pool_size=(3,3),strides=(1,1),padding='same',data_format='channels_first')(out4)
	out5_4=Conv2D(filters=128,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5_4)

	out5=concatenate(inputs=[out5_1,out5_2,out5_3],axis=1)

#	out6_1=Conv2D(filters=32,kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5)
#
#	out6_2=Conv2D(filters=32,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5)
#	out6_2=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out6_2)
#
#	out6_3=Conv2D(filters=32,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5)
#	out6_3=Conv2D(filters=128,kernel_size=(5,5),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out6_3)
#
#	out6_4=pool2(pool_size=(3,3),strides=(1,1),padding='same',data_format='channels_first')(out5)
#	out6_4=Conv2D(filters=128,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out6_4)
#
#	out6=concatenate(inputs=[out6_1,out6_2,out6_3,out6_4],axis=1)

	out7=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),
	            activation='relu')(out5)

	out8=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),
	            activation='relu')(out7)
	out8=add([out5, out8])
	out8=pool2(pool_size=(2,2),data_format='channels_first')(out8)

	out9=Flatten()(out8)


	out10=Dense(units=11,
	           kernel_initializer='normal',
               kernel_regularizer='l2',
	           activation='softmax')(out9)

	cnn = Model(inputs=input,outputs=out10)
	return cnn

#temp/resgooglenet764_v1
def createNewModel(patchSize):
	seed=5
	np.random.seed(seed)
	input=Input(shape=(1,patchSize[0, 0], patchSize[0, 1]))
	out1=Conv2D(filters=32,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(input)
	out2=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out1)
	out2=pool2(pool_size=(2,2),data_format='channels_first')(out2)

	out3=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out2)
	out4=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out3)
	out4=add([out2,out4])
	out4=pool2(pool_size=(2,2),data_format='channels_first')(out4)

	out_3=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out4)
	out_4=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out_3)

	out5_1=Conv2D(filters=64,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out_4)

	out5_2=Conv2D(filters=32,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out_4)
	out5_2=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5_2)

	out5_3=Conv2D(filters=64,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out_4)
	out5_3=Conv2D(filters=128,kernel_size=(5,5),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5_3)

	out5_4=pool2(pool_size=(3,3),strides=(1,1),padding='same',data_format='channels_first')(out4)
	out5_4=Conv2D(filters=128,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5_4)

	out5=concatenate(inputs=[out5_1,out5_2,out5_3],axis=1)

	sout6=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),
	            activation='relu')(out5)

	out7=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),
	            activation='relu')(out5)

	out8=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),
	            activation='relu')(out7)
	out8=add([sout6, out8])
	out8=pool2(pool_size=(2,2),data_format='channels_first')(out8)

	out9=Flatten()(out8)


	out10=Dense(units=11,
	           kernel_initializer='normal',
               kernel_regularizer='l2',
	           activation='softmax')(out9)

	cnn = Model(inputs=input,outputs=out10)
	return cnn


#temp/Inception-ResNet for 180180
def createNewModel(patchSize):
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


#temp/Dense ResNet for 180180
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



def createNewModel(patchSize):
	seed=5
	np.random.seed(seed)
	input=Input(shape=(1,patchSize[0, 0], patchSize[0, 1]))
	out=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(2, 2),kernel_regularizer=l2(1e-6),activation='relu')(input)
	out=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out)
	out1=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	out=Block(out1,128,with_shortcut=True)
	out=Block(out,128,with_shortcut=False)
	out=concatenate(inputs=[out1,out],axis=1)
	out2=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	out=Block(out2,128,with_shortcut=True)
	out=Block(out,128,with_shortcut=False)
	out1=pool2(pool_size=(2,2),data_format="channels_first")(out1)
	out=concatenate(inputs=[out1,out2,out],axis=1)
	out3=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	out=Block(out3,256,with_shortcut=True)
	out=Block(out,256,with_shortcut=False)
	out2=pool2(pool_size=(2,2),data_format="channels_first")(out2)
	out=concatenate(inputs=[out2,out3,out],axis=1)
	out4=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	#out5=GlobalAveragePooling2D(data_format='channels_first')(out4)

	out5=Flatten()(out4)

	out6=Dense(units=11,
	           kernel_initializer='normal',
               kernel_regularizer='l2',
	           activation='softmax')(out5)

	cnn = Model(inputs=input,outputs=out6)
	return cnn



#temp/resdensenet2 for 180180
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


def createNewModel(patchSize):
	seed=5
	np.random.seed(seed)
	input=Input(shape=(1,patchSize[0, 0], patchSize[0, 1]))
	out=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(input)
	out=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out)
	out1=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	out=Block(out1,128,with_shortcut=True)
	out=Block(out,128,with_shortcut=False)
	out=Block(out,128,with_shortcut=False)
	out=concatenate(inputs=[out1,out],axis=1)
	out2=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	out=Block(out2,128,with_shortcut=True)
	out=Block(out,128,with_shortcut=False)
	out=Block(out,128,with_shortcut=False)
	out1=pool2(pool_size=(2,2),data_format="channels_first")(out1)
	out=concatenate(inputs=[out1,out2,out],axis=1)
	out3=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	out=Block(out3,256,with_shortcut=True)
	out=Block(out,256,with_shortcut=False)
	out=Block(out,256,with_shortcut=False)
	out2=pool2(pool_size=(2,2),data_format="channels_first")(out2)
	out=concatenate(inputs=[out2,out3,out],axis=1)
	out4=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	#out5=GlobalAveragePooling2D(data_format='channels_first')(out4)

	out5=Flatten()(out4)

	out6=Dense(units=11,
	           kernel_initializer='normal',
               kernel_regularizer='l2',
	           activation='softmax')(out5)

	cnn = Model(inputs=input,outputs=out6)
	return cnn

#temp/resdensenet3 for 180180
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

def createNewModel(patchSize):
	seed=5
	np.random.seed(seed)
	input=Input(shape=(1,patchSize[0, 0], patchSize[0, 1]))
	out=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(input)
	out=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out)
	out1=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	out=Block(out1,128,with_shortcut=True)
	out=Block(out,128,with_shortcut=False)
	out=Block(out,128,with_shortcut=False)
	out=concatenate(inputs=[out1,out],axis=1)
	out2=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	out=Block(out2,128,with_shortcut=True)
	out=Block(out,128,with_shortcut=False)
	out=Block(out,128,with_shortcut=False)
	out1=pool2(pool_size=(3,3),strides=(2,2),data_format="channels_first")(out1)
	out=concatenate(inputs=[out1,out2,out],axis=1)
	out3=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	out=Block(out3,256,with_shortcut=True)
	out=Block(out,256,with_shortcut=False)
	out2=pool2(pool_size=(3,3),strides=(2,2),data_format="channels_first")(out2)
	out=concatenate(inputs=[out2,out3,out],axis=1)
	out4=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	#out5=GlobalAveragePooling2D(data_format='channels_first')(out4)

	out5=Flatten()(out4)

	out6=Dense(units=11,
	           kernel_initializer='normal',
               kernel_regularizer='l2',
	           activation='softmax')(out5)

	cnn = Model(inputs=input,outputs=out6)
	return cnn






#temp/resnet6c2pfor 4040
def createNewModel(patchSize):
	seed=5
	np.random.seed(seed)
	input=Input(shape=(1,patchSize[0, 0], patchSize[0, 1]))
	out1=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(input)
	out2=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out1)

	out3=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out2)
	out4=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out3)
	out4=pool2(pool_size=(2,2),data_format='channels_first')(out4)

	sout5=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out4)
	out5=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out4)
	out6=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5)
	out6=add([sout5,out6])
	out6=pool2(pool_size=(2,2),data_format='channels_first')(out6)

	out10=Flatten()(out6)


	out11=Dense(units=11,
	           kernel_initializer='normal',
               kernel_regularizer='l2',
	           activation='softmax')(out10)

	cnn = Model(inputs=input,outputs=out11)
	return cnn


#temp/resnet6c2p_v1 for 4040
def createNewModel(patchSize):
	seed=5
	np.random.seed(seed)
	input=Input(shape=(1,patchSize[0, 0], patchSize[0, 1]))
	out1=Conv2D(filters=32,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(input)
	out2=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out1)

	sout3=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out2)
	out3=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out2)
	out4=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out3)
	out4=add([sout3,out4])
	out4=pool2(pool_size=(2,2),data_format='channels_first')(out4)

	sout5=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out4)
	out5=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out4)
	out6=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5)
	out6=add([sout5,out6])
	out6=pool2(pool_size=(2,2),data_format='channels_first')(out6)

	out10=Flatten()(out6)


	out11=Dense(units=11,
	           kernel_initializer='normal',
               kernel_regularizer='l2',
	           activation='softmax')(out10)

	cnn = Model(inputs=input,outputs=out11)
	return cnn


#temp/resnet6c2p_v2 for 4040
def createNewModel(patchSize):
	seed=5
	np.random.seed(seed)
	input=Input(shape=(1,patchSize[0, 0], patchSize[0, 1]))
	out1=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(input)
	out2=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='valid',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out1)

	sout3=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out2)
	out3=Conv2D(filters=64,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out2)
	out4=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out3)
	out4=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out4)
	out4=add([sout3,out4])
	out4=pool2(pool_size=(2,2),data_format='channels_first')(out4)

	sout5=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out4)
	out5=Conv2D(filters=64,kernel_size=(1,1),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out4)
	out6=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out5)
	out6=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out6)
	out6=add([sout5,out6])
	out6=pool2(pool_size=(2,2),data_format='channels_first')(out6)

	out10=Flatten()(out6)


	out11=Dense(units=11,
	           kernel_initializer='normal',
               kernel_regularizer='l2',
	           activation='softmax')(out10)

	cnn = Model(inputs=input,outputs=out11)
	return cnn

#temp/resdensenet1 for 4040
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



def createNewModel(patchSize):
	seed=5
	np.random.seed(seed)
	input=Input(shape=(1,patchSize[0, 0], patchSize[0, 1]))
	out=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(2, 2),kernel_regularizer=l2(1e-6),activation='relu')(input)
	out=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out)
	out1=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	out=Block(out1,128,with_shortcut=True)
	out=Block(out,128,with_shortcut=False)
	out=concatenate(inputs=[out1,out],axis=1)
	out2=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	out=Block(out2,256,with_shortcut=True)
	out=Block(out,256,with_shortcut=False)
	out1=pool2(pool_size=(2,2),data_format="channels_first")(out1)
	out=concatenate(inputs=[out1,out2,out],axis=1)
	out3=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	#out5=GlobalAveragePooling2D(data_format='channels_first')(out4)

	out5=Flatten()(out3)

	out6=Dense(units=11,
	           kernel_initializer='normal',
               kernel_regularizer='l2',
	           activation='softmax')(out5)

	cnn = Model(inputs=input,outputs=out6)
	return cnn

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



def createNewModel(patchSize):
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


#temp/resdensenet3 for 4040
def Block(input,num_filters,with_shortcut):
	out1 = Conv2D(filters=num_filters/4, kernel_size=(1, 1), kernel_initializer='he_normal', weights=None, padding='same',
	              strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(input)
	out2 = Conv2D(filters=num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None, padding='same',
	              strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(out1)
	out3 = Conv2D(filters=num_filters, kernel_size=(1, 1), kernel_initializer='he_normal', weights=None, padding='same',
	              strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(out2)

	if with_shortcut:
		input = Conv2D(filters=num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None,
		              padding='same',strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(input)
		return add([input,out3])
	else:
		input = Conv2D(filters=num_filters, kernel_size=(1, 1), kernel_initializer='he_normal', weights=None,
		              padding='same',strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')(input)
		return add([input,out3])



def createNewModel(patchSize):
	seed=5
	np.random.seed(seed)
	input=Input(shape=(1,patchSize[0, 0], patchSize[0, 1]))
	out=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(input)
	out1=Conv2D(filters=64,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out)
	#out1=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	sout1=Conv2D(filters=128,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out1)
	out=Block(out1,128,with_shortcut=True)
	out=Block(out,128,with_shortcut=False)
	out=add([sout1,out])
	out=concatenate(inputs=[out1,out],axis=1)
	out2=pool2(pool_size=(3,3),strides=(2,2),data_format='channels_first')(out)

	sout2=Conv2D(filters=256,kernel_size=(3,3),kernel_initializer='he_normal',weights=None,padding='same',strides=(1, 1),kernel_regularizer=l2(1e-6),activation='relu')(out2)
	out=Block(out2,256,with_shortcut=True)
	out=Block(out,256,with_shortcut=False)
	out=add([sout2,out])
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







