import argparse
import os.path
import shutil
# set GPU

#insert argparser
# input parsing
from config.PATH import GPU_HOME_PATH


def fSetGPU():
    parser = argparse.ArgumentParser(description='''setting a GPU, replace the 's1222' in the code with your own number, you should have a '.theanorc' textfile in your home directory, which contains the theano config. 
    additionally you should have 4 other textfiles called 'theanorc_gpuX' with the corresponding GPU set in ''',
                                     epilog='''by Marvin Jandt''')
    parser.add_argument('-g', '--GPU', nargs=1, type=int, help='desired GPU', default=3)
    args = parser.parse_args()
    iGPU = args.GPU              #[0]
    sHomedir = GPU_HOME_PATH
    print(sHomedir + 'theanorc_GPU' + str(iGPU))
    if (os.path.isfile(sHomedir + 'theanorc_GPU' + str(iGPU))):
        shutil.copyfile(sHomedir + 'theanorc_GPU' + str(iGPU), sHomedir + '.theanorc');
        print('Switching GPU -> ' + str(iGPU))
    else:
        print('something did not work. the file the file theanorc_GPU' + str(iGPU) + ' may not exist')