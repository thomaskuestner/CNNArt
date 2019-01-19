# This file is for fetching the file paths for the NAKO data set
import os
import pandas as pd
from random import shuffle

# the  sPathToExcel could be loaded from the configuration file

def get_train_eval_files (data_dir, pattern, train_eval_ratio = 0.8,
                          start_id = 100000, end_id = 190000, sPathToExcel =""):
    # choose those subject
    subject_ids = [i for i in range(start_id, end_id)]
    subject_path_id = [str(id) + '_30' for id in subject_ids]
    subject_paths = [os.path.join(data_dir,subject_folder) for subject_folder in subject_path_id]

    # check the existing subject
    path_collections = []
    for subject_path in subject_paths:
        if os.path.exists(subject_path):
            path_collections.append(subject_path)
    #print(path_collections)

    # check the existing pattern image
    path_collections = [os.path.join(path, 'image') for path in path_collections]
    #print(path_collections)

    data_path = []
    for item in path_collections:
        files = os.listdir(item)
        for file in files:
            #print(file)
            if file.endswith(pattern):
                data_path.append(os.path.join(item, file))
                break
    print('train_paths: ', data_path)

    chose_subjects = [path.split('/')[6] for path in data_path]
    qualityInfo = pd.read_excel(os.path.join(sPathToExcel, 'CombiExcel.xlsx'))
    qualityInfo = qualityInfo.rename(index=str, columns={'PatientID_FolderName': 'Patient ID'})

    #print(qualityInfo)
    #print(qualityInfo[qualityInfo['Patient ID'] == '102597_30']['QualityRating'])

    label_list = [qualityInfo[qualityInfo['Patient ID'] == id]['QualityRating'].values[0] for id in chose_subjects]
    print('labels ', label_list)

    count_1, count_2, count_3 = 0, 0, 0

    for item in label_list:
        if item == 1:
            count_1 += 1
        elif item == 2:
            count_2 += 1
        elif item == 3:
            count_3 += 1
    print('To have a general idea about label balance:  the number of label 1: ', count_1 ,' label 2: ', count_2, ' label 3: ', count_3)

    shuffle(data_path)

    pos = int(len(data_path) * train_eval_ratio)

    training_path = data_path[:pos]
    chose_subjects = [path.split('/')[6] for path in training_path]
    train_labels = [qualityInfo[qualityInfo['Patient ID'] == id]['QualityRating'].values[0] for id in chose_subjects]
    train_labels = [ x-1 for x in train_labels]

    eval_path = data_path[pos:]
    chose_subjects = [path.split('/')[6] for path in eval_path]
    eval_labels = [qualityInfo[qualityInfo['Patient ID'] == id]['QualityRating'].values[0] for id in chose_subjects]
    eval_labels = [x-1 for x in eval_labels]

    print('train paths: ', training_path)
    print('train labels ', train_labels)
    print('eval paths ', eval_path)
    print('eval labels ', eval_labels)
    return training_path, train_labels, eval_path, eval_labels


if __name__ == '__main__':
    path = '/home/d1274/no_backup/d1274/NAKO_tf'
    start_id = 100000
    end_id = 190000
    get_train_eval_files(path, start_id=start_id, end_id=end_id, pattern='_F.tfrecord')