# https://github.com/thomaskuestner/CNNArt
from setuptools import setup
import sys

def readme():
    with open('README.md') as f:
        return f.read()

if sys.argv[-1] == 'test':
    setup(name='CNNArt',
          version='1.0',
          description='M R artifact detection',
          long_description=readme(),
          classifiers=[
            'Development Status :: Beta',
            'License :: Apache License 2.0',
            'Programming Language :: Python :: 3.6',
            'Topic :: CNN',
          ],
          keywords='CNN, artifacts, motion, multiclass',
          url='https://github.com/thomaskuestner/CNNArt',
          author='kSpace Astronauts',
          author_email='thomas.kuestner@iss.uni-stuttgart.de',
          license='Apache2',
          setup_requires=['pytest-runner'],
          tests_require=['pytest'],
          include_package_data=True,
          zip_safe=False,
          install_requires=['jsonschema==2.6.0', 'pickleshare==0.7.5', 'h5py==2.8.0', 'keras==2.1.6',
                            'matplotlib==2.2.3', 'pandas==0.23.4', 'scipy==1.1.0', 'protobuf==3.6.0',
                            'tensorflow==1.12.0', 'pyYAML==3.13', 'PyQt5-stubs==5.11.3.3', 'PyQt5==5.11.3',
                            'numpy==1.14.5', 'pandas==0.23.4', 'argparse==1.4.0', 'hyperas==0.4', 'hyperopt==0.1.1',
                            'dicom==0.9.9', 'pydicom==1.2.0', 'dicom-numpy==0.1.4', 'pydot==1.3.0',
                            'pyqtgraph==0.10.0', 'python-csv==0.0.11', 'gtabview==0.8', 'pyqtdeploy==2.3.1',
                            'nibabel==2.3.3', 'Pillow==5.3.0', 'xtract==0.1a3',
                            'scikit-image==0.14.1', 'scikit-learn==0.19.2', 'seaborn==0.9.0'])
    sys.exit()
