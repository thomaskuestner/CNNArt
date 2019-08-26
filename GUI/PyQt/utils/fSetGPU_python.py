import argparse
import os.path
import shutil
# set GPU

#insert argparser
# input parsing
parser = argparse.ArgumentParser(description='''setting a GPU, replace the 'sXXXX' in the code with your own number, you should have a '.theanorc' textfile in your home directory, which contains the theano config. 
additionally you should have 4 other textfiles called 'theanorc_gpuX' with the corresponding GPU set in ''', epilog='''kSpace astronauts''')
parser.add_argument('-g','--GPU', nargs = 1, type= int, help='desired GPU', default=0)
args=parser.parse_args()
iGPU=args.GPU[0]
sHomedir='/home/sXXXX/'

if(os.path.isfile(sHomedir+'theanorc_gpu'+ str(iGPU))):
    shutil.copyfile(sHomedir+'theanorc_gpu'+ str(iGPU), sHomedir+'.theanorc');
    print ('Switching GPU -> '+ str(iGPU))
else:
    print ('something did not work. the file the file theanorc_gpu'+ str(iGPU) +' may not exist')


