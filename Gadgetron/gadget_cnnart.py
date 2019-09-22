import numpy as np
from gadgetron import Gadget
import pickle
import sys, os
sys.path.append(os.getcwd()) 
sys.path.append('/opt/data/CNNArt')
from Gadgetron.fast_test_call_cnnart import gadget_cnnart
# from Gadgetron.cnnart_for_gadgetron import gadget_cnnart
import time
import h5py
# from keras.models import model_from_json

class GadgetronCNNArt(Gadget):
    def process(self, head, data):

        # For Test
        # timestamp = str(int(time.time()*1e3))
        # if not os.path.exists('/opt/data/gadgetron/testdata/'):
        #     os.makedirs('/opt/data/gadgetron/testdata/')
        #     os.makedirs('/opt/data/gadgetron/testdata/data/')
        #     os.makedirs('/opt/data/gadgetron/testdata/head/')
        # np.save('/opt/data/gadgetron/testdata/data/data_in_'+timestamp+'.npy', data)
        # with open('/opt/data/gadgetron/testdata/head/head_in_'+timestamp+'.pickle', 'wb') as f:
        #     pickle.dump(head, f)

        print('==== Gadget start ====')
        prediction = gadget_cnnart(data)
        # print('====', data.shape, '====')
        # prediction = np.load('unpatched_img_background.npy')
        data = prediction/prediction.max()*data.max()
        self.put_next(head, data)
        return 0