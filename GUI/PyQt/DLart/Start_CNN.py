import os

from DLart.cnn_main import CNN_execute
from config.PATH import TEST_PATH, CNN_PATH

PathIn = TEST_PATH + os.sep + "Training and Test data/normal/AllData_Move_05_label05_val_ab4040.h5"
PathOut = CNN_PATH
learning_rate = 0.001
batchSize = 64
epoch = 100
CNN_execute(PathIn, PathOut, batchSize, learning_rate, epoch, 'motion_head')