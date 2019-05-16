import os
import tensorflow as tf
from utils.tfrecord.training.multiclass_3D_CNN import buildModel
from utils.tfrecord.training.create_dataset import create_dataset
#from create_dataset_copy import create_dataset

config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config = config))


if __name__ == '__main__':

    print('begin creating the input dataset')
    data_dir = '/home/d1274/no_backup/d1274/data'

    datagen= create_dataset(data_dir)


    # The image shape is (236, 320, 260)
    #model = Resnet3DBuilder.build_resnet_50((64, 64, 64, 1), 3)
    model = buildModel((64, 64, 64, 1), 3)
    print(model.summary())

    model.compile(loss="categorical_crossentropy",
                  optimizer="sgd",
                  metrics=['acc'])

    model_dir = os.path.join('/home/d1274/no_backup/d1274/model', "IQA_test_with_tfrecord")
    os.makedirs(model_dir, exist_ok=True)
    print("model_dir: ",model_dir)
    est_iqa = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                    model_dir=model_dir)

    input_name = model.input_names[0]
    print(input_name)
    # model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])


    # attention: steps_per_epoch = num_samples * num_patches_per_image/ batch_size

    model.fit_generator(
        datagen,
        steps_per_epoch = 240,
        epochs=500)









