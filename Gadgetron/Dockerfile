# $ docker build -t cnnart_gadgetron .
# FROM gadgetron/ubuntu_1804_cuda92_cudnn7:latest
FROM gadgetron/ubuntu_1804_no_cuda:latest
# RUN rm /usr/bin/python && cp /usr/bin/python3 /usr/bin/python && cp /usr/bin/pip3 /usr/bin/pip 
RUN  mkdir /opt/data && cd /opt/data && git clone https://github.com/thomaskuestner/CNNArt.git \
    && cd CNNArt && git checkout yang_liu && git pull
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2 \
    && pip install --upgrade pip \
    && pip install -r /opt/data/CNNArt/requirements.txt \
    && pip install scikit-image pydicom dicom-numpy scikit-learn pyyaml h5py tensorflow keras matplotlib theano\
    && pip uninstall -y h5py && pip install --no-binary=h5py h5py \
    && mkdir ~/.keras && cp /opt/data/CNNArt/Gadgetron/keras.json ~/.keras \
    && cp /opt/data/CNNArt/Gadgetron/cnnart_h5.xml /usr/local/share/gadgetron/config

# manual move cnnart_trainednets to /opt/CNNArt
#WORKDIR /opt/data

CMD /bin/bash

# cd to work direction
# $ docker run -t --name gt1 --detach --volume $(pwd):/opt/data cnnart_gadgetron
# $ docker exec -ti gt1 /bin/bash
# in docker: $ ismrmrd_generate_cartesian_shepp_logan
# in docker: $ gadgetron_ismrmrd_client -f testdata.h5 -o out.h5 -c cnnart.xml

# next run:
# $ docker start gt1
# $ docker exec -ti gt1 bash

