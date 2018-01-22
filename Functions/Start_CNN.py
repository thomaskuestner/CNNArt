from cnn_main import*

PathIn = "/home/d1224/no_backup/d1224/PatchbasedLabeling Results/Training and Test data/normal/AllData_Move_05_label05_val_ab4040.h5"
PathOut = "/home/d1224/no_backup/d1224/PatchbasedLabeling Results/linse3/"
learning_rate = 0.001
batchSize = 64
epoch = 100
CNN_execute(PathIn, PathOut, batchSize, learning_rate, epoch, 'motion_head')