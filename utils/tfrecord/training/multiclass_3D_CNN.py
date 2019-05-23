from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from tensorflow.keras.layers import (
    Conv3D,
    AveragePooling3D,
    MaxPooling3D
)

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

def _handle_data_format():
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4

def buildModel(patchShape, numClasses):
    input = Input(shape=patchShape)
    n_base_fileter = 32
    _handle_data_format()
    conv = Conv3D(filters=n_base_fileter, kernel_size=(7, 7, 7),
                  strides=(2, 2, 2), kernel_initializer="he_normal",
                  )(input)
    norm = BatchNormalization(axis=CHANNEL_AXIS)(conv)
    conv1 = Activation("relu")(norm)
    pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2),
                         padding="same")(conv1)
    flatten1 = Flatten()(pool1)
    dense = Dense(units=numClasses,
                  kernel_initializer="he_normal",
                  activation="softmax",
                  kernel_regularizer=l2(1e-4))(flatten1)
    model = Model(inputs=input, outputs=dense)
    return model
