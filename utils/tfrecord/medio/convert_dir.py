#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file should be re defined accoirding to the situation of the directory !

from utils.tfrecord.medio import convert_tf, parse_gen , read_image
import pathlib as pl
import os
import logging


def dir2tf(dir_data, dir_tf, b_verbose=False, b_skip_existing=False):

    # convert to paths
    if not isinstance(dir_data, pl.Path):
        path_data = pl.Path(dir_data)
    else:
        path_data = dir_data

    if not isinstance(dir_tf, pl.Path):
        path_tf = pl.Path(dir_tf)
    else:
        path_tf = dir_tf

    path_tf.mkdir(parents=True, exist_ok=True)
    subjects_existing = [x for x in os.listdir(path_tf) if os.path.isdir(path_tf.joinpath(x))]

    for group, subject_path in parse_gen.parse_data_gen(path_data):

        # skip existing subjects
        if b_skip_existing:
            if any([subject_path.name == x for x in subjects_existing]):
                continue

        if b_verbose:
            print('group: ', group )
            print('subject path: ', subject_path)


        # for the tfrecordm what we want to save is like Q1/...tfrecord
        #path_temp = path_tf.joinpath(group + '/' + subject_path.name)
        #print('path temp: ', path_temp)
        #path_temp.mkdir(parents=True, exist_ok=True)

        image_shapes = []
        expected_shape = (236, 320, 260)

        image = read_image.read_mat_image(subject_path)  # image = numpy array
        #print('image_shape: ', image.shape)

        # skip files not conforming minimal shape requirements
        if image.shape != expected_shape:
            logging.warning('skipped file %s: size not correct' % subject_path)
            continue

        # fetch shape to sort out bad nii files
        image_shapes.append(image.shape)

        path_file =  path_tf.joinpath(group + '/' + subject_path.name.split('.')[0] + '.tfrecord')

        print('creating ', path_file)
        path_file.parent.mkdir(parents=True, exist_ok=True)
        convert_tf.im2tfrecord(image=image, path=path_file)

