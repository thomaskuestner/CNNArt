import os
import os.path
import csv
from MRData import MRData
from random import shuffle


class DatabaseInfo:

    sPathOut = '' # set output path for results
    sDatabase = ''
    sSubDirs = ''
    sPathIn = ''
    lPats = ''
    lImgData = '' # list of imaging data

    def __init__(self, sDatabase = 'MRPhysics', sSubDirs = ['newProtocol','dicom_sorted','testout'], sDatabaseRootPath = '/med_data/ImageSimilarity/Databases', *args):
        self.sDatabaseRootPath = sDatabaseRootPath
        self.sDatabase = sDatabase
        self.sSubDirs = sSubDirs # name of subdirectory in [database, patient]

        if not self.sSubDirs[0]:   
            self.sPathIn = sDatabaseRootPath + os.sep + self.sDatabase
        else:
            self.sPathIn = sDatabaseRootPath + os.sep + self.sDatabase + os.sep + self.sSubDirs[0]
        # parse patients

        self.lPats = [name for name in os.listdir(self.sPathIn) if os.path.isdir(os.path.join(self.sPathIn, name))]

        # parse config file (according to set database) => TODO (TK): replace by automatic parsing in directory

        # if run the file which is under a folder like deepvis
        # need to change the path back to 'CNNArt'
        # otherwise the file cannot find the content under folder 'config'

        if len(args) == 0:
            ifile = open('config'+os.sep+'database'+os.sep+self.sDatabase+'.csv',"r") # config file must exist for database!
        else:
            ifile = open(args[0]+os.sep+'config' + os.sep + 'database' + os.sep + self.sDatabase + '.csv', "r")

        reader = csv.reader(ifile)
        next(reader)
        #lImgData = []
        #for i, rows in enumerate(reader):
        #    if i == 0:
        #        continue
        #    self.lImgData.append(MRData(rows[0],rows[1],rows[2],rows[3]))
        self.lImgData = [MRData(rows[0],rows[1],rows[2],rows[3]) for rows in reader]

        ifile.close()

    def get_mrt_model(self):
        #{key: value for key, value in A.__dict__.items() if not key.startswith('__') and not callable(key)}
        return {item.sPath:item.sNumber for item in self.lImgData}

    def get_mrt_smodel(self):
        return {item.sPath: item.sBodyregion for item in self.lImgData}

    def get_mrt_artefact(self):
        return {item.sPath: item.sArtefact for item in self.lImgData}



class NAKOInfo(DatabaseInfo):

    #  parse label according to the file name
    def parse_file_name(self, fileName):
        if fileName.find('_deep_') != -1:
            return 2
        elif fileName.find('bh') != -1:
            return 0
        elif fileName.find('fb') != -1:
            return 1

    def parse_label_list (self, file_list):
        label_list = []
        for file in file_list:
            label_list.append(self.parse_file_name(file))
        return label_list

    # getting the filename list according to the specific pattern and if it is testing file or train file
    def get_file_lists(self, pattern='_F_', test_groups = ['Q10', 'Q11', 'Q12'], is_testing = True ):

        data_dir = self.sDatabaseRootPath + os.sep + self.sDatabase
        file_lists = []
        groups = os.listdir(data_dir)

        for group in groups:
            # check if the group belong to the test group or not

            if (not is_testing) and (group in test_groups):
                continue
            if (is_testing) and (group not in test_groups):
                continue
            group_path = os.path.join(data_dir, group)
            files = os.listdir(group_path)

            for file in files:
                if file.endswith('.tfrecord') and pattern in file:
                    file_lists.append(os.path.join(group_path, file))

        return  file_lists


    def get_train_eval_files(self, pattern='_F_', test_groups = ['Q10', 'Q11', 'Q12'],
                             train_eval_ratio = 0.85, lShuffleTraining = True):

        file_list = self.get_file_lists(pattern=pattern, test_groups=test_groups, is_testing=False)

        num_train_files = int(len(file_list) * train_eval_ratio)
        if lShuffleTraining :
            shuffle(file_list)
        train_files = file_list[:num_train_files]
        eval_files = file_list[num_train_files:]

        train_labels = self.parse_label_list(train_files)
        eval_labels = self.parse_label_list(eval_files)

        return train_files, train_labels, eval_files, eval_labels

    def get_test_files(self, pattern='_F_', test_groups = ['Q10', 'Q11', 'Q12']):

        test_files = self.get_file_lists(pattern=pattern, test_groups=test_groups, is_testing= True)

        test_labels = self.parse_label_list(test_files)

        return test_files, test_labels