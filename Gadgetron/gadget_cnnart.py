import numpy as np
from gadgetron import Gadget
import pydicom
import sys
sys.path.append('/opt/data')
import pickle
import sys, os
sys.path.append(os.getcwd()) 
sys.path.append('/opt/data/CNNArt')
from Gadgetron.fast_test_call_cnnart import gadget_cnnart
# from Gadgetron.cnnart_for_gadgetron import gadget_cnnart
import time
import os
from keras.models import model_from_json

class GadgetronCNNArt(Gadget):
    def process(self, head, data):
        timestamp = str(int(time.time()*1e3))
        if not os.path.exists('/opt/data/gadgetron/testdata/'):
            os.makedirs('/opt/data/gadgetron/testdata/')
            os.makedirs('/opt/data/gadgetron/testdata/data/')
            os.makedirs('/opt/data/gadgetron/testdata/head/')
        np.save('/opt/data/gadgetron/testdata/data/data_in_'+timestamp+'.npy', data)
        with open('/opt/data/gadgetron/testdata/head/head_in_'+timestamp+'.pickle', 'wb') as f:
            pickle.dump(head, f)

with open('/opt/data/cnnart_trainednets/motion/FCN/FCN 3D-VResFCN-Upsampling final Motion Binary_3D_128x128x16_2019-03-28_18-46.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('/opt/data/cnnart_trainednets/motion/FCN/FCN 3D-VResFCN-Upsampling final Motion Binary_3D_128x128x16_2019-03-28_18-46_weights.h5')
        print('==== Gadget start ====')
        prediction = gadget_cnnart(data)
        data = prediction/prediction.max()*data.max()
        # patch_size, overlap_rate = [128, 128, 16], 0.4
        # if len(data.shape) == 4:  # for 3D
        #     # data's constellation is x, y, z, 1
        #     data2 = np.swapaxes(data[:, :, :, 0], 0, 2)
        #     # data2 is z, y, x
        #     rgb_data = gadget_cnnart(data2, patch_size, overlap_rate)
        #     # rgb_data is z, rgb, y, x
        #     rgb_data = np.transpose(rgb_data)
        #     # now rgb_data is x, y, rgb, z
        #     head.matrix_size[2] = 3
        #     np.save('rgb_data.npy', rgb_data)
        #     with open('head.pickle', 'wb') as f:
        #         pickle.dump(head, f)
        #     self.put_next(head,rgb_data)

        # elif len(data.shape) == 3:  # for 2D
        #     data2 = gadget_cnnart(data, patch_size, overlap_rate )
        #     head.matrix_size[2] = 3
        #     self.put_next(head, data2)
        self.put_next(head, data)
        return 0