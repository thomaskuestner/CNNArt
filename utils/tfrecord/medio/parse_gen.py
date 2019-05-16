#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is intended for NAKO_IQA data_set
# To generate the mat data

import pathlib
# use glob / fnmatch / os.scandir for folder parsing


def parse_data_gen(data_path, b_verbose=False):
    """ Parses the given folder path for subjects and yields incorporated medical files """

    if not isinstance(data_path, pathlib.Path):
        path = pathlib.Path(data_path)
    else:
        path = data_path

    for group in path.iterdir():
        if group.is_dir():

            # img_dir shall refers to the layer dicom_sorted
            for img_dir in group.iterdir():
                if img_dir.is_dir():

                    # This shall refer to the folder like 3D_GRE_TRA_bh_F_0006
                    for subject in img_dir.iterdir():
                        if subject.is_file() and str(subject).endswith('.mat'):


                            yield group.name, subject


if __name__ == '__main__':

    data_path = '/med_data/NAKO/NAKO_IQA'
    for group, subject  in parse_data_gen(data_path):
        print('group: ', group)
        print('subject: ', subject)