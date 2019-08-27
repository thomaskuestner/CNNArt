import numpy as np
from gadgetron import Gadget
import pydicom
import sys
sys.path.append('/opt/data')
import pickle
from CNNArt.Gadgetron.fast_test_call_cnnart import gadget_cnnart
#from CNNArt.Gadgetron.cnnart_for_gadgetron import gadget_cnnart
import time
import os

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