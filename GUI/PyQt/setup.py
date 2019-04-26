# https://github.com/thomaskuestner/CNNArt
from setuptools import setup
import sys


def readme():
    with open('README.md') as f:
        return f.read()


#if sys.argv[-1] == 'test':
if __name__ == "__main__":
    setup(name='imagine',
          version='1.0',
          description='imagine visualization GUI',
          long_description=readme(),
          classifiers=[ 
            'Development Status :: Beta',
            'License :: Apache License 2.0',
            'Programming Language :: Python :: 3.6',
            'Topic :: CNN',
          ],
          keywords='2D/3D/4D/5D image visualization, CNN, artifacts, motion, multiclass',
          url='https://github.com/thomaskuestner/CNNArt',
          author='kSpace Astronauts',
          author_email='thomas.kuestner@iss.uni-stuttgart.de',
          license='Apache2',
          setup_requires=['pytest-runner'],
          tests_require=['pytest'],
          include_package_data=True,
          zip_safe=False,
          install_requires=['jsonschema', 'pickleshare', 'h5py', 'keras', 'matplotlib',
                            'pandas', 'scipy', 'protobuf', 'tensorflow', 'pyYAML', 'PyQt5-stubs', 'PyQt5',
                            'numpy', 'pandas', 'argparse', 'hyperas', 'hyperopt', 'graphviz', 'dicom',
                            'pydicom', 'dicom-numpy', 'pyqtgraph', 'python-csv', 'gtabview', 'pyqtdeploy',
                            'nibabel', 'Pillow', 'pydot', 'xtract', 'scikit-image', 'scikit-learn', 'seaborn'])
    #sys.exit()
