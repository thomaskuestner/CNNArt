#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# create the tensorflow folder from the original image folder

from medio import convert_dir

if __name__ == '__main__':

    path_tf = '/home/d1274/no_backup/d1274/data'
    path_data = '/med_data/NAKO/NAKO_IQA'

    convert_dir.dir2tf(path_data, path_tf, b_verbose=True, b_skip_existing=False)
