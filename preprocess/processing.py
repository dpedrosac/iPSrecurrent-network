#!/usr/bin/env python
# -*- coding: utf-8 -*-

import getpass
import glob
import os
import warnings
import itertools
import re

import numpy as np
import pandas as pds
import scipy
import yaml
from keras.utils import to_categorical
from dependencies import ROOTDIR, FILEDIR
from numpy.random import default_rng
import preprocess.HelperFunctions as helpers

with open(os.path.join(ROOTDIR, 'config.yaml'), 'r') as f:
    CONFIGDATA = yaml.load(f.read(), Loader=yaml.FullLoader)
CONFIGDATA = CONFIGDATA[0]

output = helpers.Output()
class DataProcessing:
    def __init__(self):
        self.debug = False
        # self.generate_subjlist(filename_patlist='', ignore_bad=CONFIGDATA[0]['dataworks']['ignbad'])

    @staticmethod
    def generate_subjlist(filename_patlist, pseud='', ignore_bad=True):
        """imports pseudonyms of subj. to be processed; this is necessary to read data later"""

        if not filename_patlist:
            warnings.warn("No filename for the patient list was defined. Stopping script here", category=Warning)
            return

        if not pseud:
            print("\tExtracting file pseudonyms of all subjects provided in the metadata list.")
        else:
            print("\tExtracting subj: {} for further processing.".format(pseud))

        dataframe_metadata = pds.read_excel(filename_patlist, sheet_name='working') # imports data from excel file

        subject_list, idx_list = [[] for _ in range(2)]
        if not pseud: # ignores data categorised as wrong/bad (cf. patienten_onoff.xls for details)
            if ignore_bad:
                subject_list = dataframe_metadata.pseud[dataframe_metadata.group == 1]
                idx_list = np.where(dataframe_metadata['group'] == 1)
            else:
                subject_list = dataframe_metadata.pseud
        else:
            [subject_list.append(dataframe_metadata[dataframe_metadata.pseud == x] for x in pseud)]
            [idx_list.append(np.where[dataframe_metadata.pseud == x] for x in pseud)]
            # subject_list = frame_name[frame_name.pseud == pseud]
            # idx_list = np.where(frame_name['pseud'] == pseud)

        if pseud and len(subject_list) != len(pseud):
            warnings.warn("something went wrong as len(subject_list) != len(pseud) [input]", category=Warning)

        return subject_list, idx_list, dataframe_metadata

    def concatenate_recordingsMOD(self, subject_list, conditions, devices, tasks, dataframe_metadata,
                                  window=False, scaling=False):
        """loads available data for listed subjects; at the end, there should be a dict with a trial x time x sensor
         arrangement each (representing both conditions). It may include only ACC (size = 3), ACC+GYRO (size = 6) or
         ACC+GYRO+EMG (size=14) bit also any other combination"""

        print("\tLoading all available data {} ".format('with scaling 'if scaling else 'without_scaling'))

        data = {k: [] for k in conditions}
        details = {k: [] for k in conditions}
        list_files = {k: [] for k in conditions}
        possible_combinations = list(itertools.product(subject_list, tasks, devices))
        for cond in conditions: # loop through ON- and OFF-conditions
            files_temp, loaded_temp = [[] for _ in range(2)]
            print("\t...extracting available filenames for the {} condition\n".format(cond), end='')
            files_temp.append([self.file_browser(term2searchfor=sbj + '_' + tsk + '_' + cond + '_' + dev,
                                        datpath=CONFIGDATA['dataworks']['folders'][getpass.getuser()]['datpath'])
             for sbj, tsk, dev in possible_combinations])
            print("\t\t...DONE!")

            list_files[cond] = list(itertools.chain.from_iterable(itertools.chain.from_iterable(files_temp)))

            if self.debug: # check for duplicates in the list
                import collections
                print([item for item, count in collections.Counter(list_files[cond]).items() if count > 1])

            datatemp, details_temp = [[] for _ in range(2)]
            print("\t...loading data to workspace")
            output.printProgressBar(0, len(list_files[cond]), prefix="\t\tProgress:", suffix='Complete', length=50)
            for ind_file, filename in enumerate(list_files[cond]):
                output.printProgressBar(ind_file+1, len(list_files[cond]), prefix="\t\tProgress:", suffix='Complete', length=50)
                singledata = self.load_file(os.path.join(FILEDIR, filename), scaling, debug=False)
                datatemp.append(singledata)
                pseudonym = '_'.join(os.path.split(filename)[1].split('_')[0:3])
                task = os.path.split(filename)[1].split('_')[3]
                trial_number = int(re.search(r'(trial)(\d+)', os.path.split(filename)[1]).group(2))
                device = os.path.split(filename)[1].split('_')[5]
                idx_pseud = [i for i, x in enumerate(dataframe_metadata.pseud.str.contains(pseudonym)) if x]
                details_temp.append([pseudonym, cond, task, trial_number,
                                     dataframe_metadata['age'][idx_pseud].values[0],
                                     dataframe_metadata['gender'][idx_pseud].values[0],
                                     dataframe_metadata['updrs_off'][idx_pseud].values[0],
                                     dataframe_metadata['updrs_on'][idx_pseud].values[0],
                                     dataframe_metadata['updrs_diff'][idx_pseud].values[0],
                                     dataframe_metadata['ledd'][idx_pseud].values[0],
                                     device])

            channel_idx = ['x', 'y', 'z'] # get all data from three channels accelerometer directions
            data[cond] = {ch: [] for ch in channel_idx}
            for idx_channel, ch in enumerate(channel_idx):
                data[cond][ch] = np.stack(datatemp)[:,:,idx_channel]

            details[cond] = pds.DataFrame(data=details_temp,
                                        columns=['name', 'condition', 'task', 'trial', 'age', 'gender',
                                               'updrsOFF', 'updrsON', 'updrsDiff', 'ledd', 'device'])

        return data, details

    def sliding_window(self, seq, winsize, step=1):
        """ Returns generator iterating through entire input."""

        # Verify the inputs
        try: it = iter(seq)
        except TypeError:
            raise Exception("Please make sure input is iterable.")
        if not ((type(winsize) == type(0)) and (type(step) == type(0))):
            raise Exception("Size of windows (winsize) and step must be integers.")
        if step > winsize:
            raise Exception("Steps may never be larger than winSize.")
        if winsize > len(seq):
            raise Exception("winsize must not be larger than sequence length.")

        # Pre-compute number of chunks to emit
        numchunks = ((len(seq)-winsize)/step)+1

        # Generare chunks of data
        for i in range(0,numchunks*step,step):
            yield seq[i:i+winsize]

    def load_file(self, filename, scaling=False, debug=False):
        """ helper function that reads data and returns the values into a dataframe; there are two options:
        1)  True: Mean = 0 and Std = 1
        2)  False: Mean = 0, Std remains """
        from sklearn import preprocessing
        dataframe = pds.read_table(filename, header=None, sep='\s+') #  read data from txt-file as processed via MATLAB

        # plot only if debug option is set.
        if debug:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(121).plot(dataframe)

        if scaling:
            if debug:
                plt.subplot(122).plot(preprocessing.robust_scale(dataframe.values)).show()
            return preprocessing.robust_scale(dataframe.values)
        else:
            return dataframe.values

    @staticmethod
    def file_browser(term2searchfor, datpath=''):
        """function which helps to find files with certain content, specified in the input"""

        file = []
        if not datpath:
            datpath = os.path.join(FILEDIR, 'raw_data')

        [file.append(x) for x in glob.glob(os.path.join(datpath) + '*') if term2searchfor in x]

        return sorted(file, key=lambda x: float(re.split('([0-9]+)', x)[-2]))

    @staticmethod
    def arrange_data(dattemp, datlength):
        """helper function that interpolates data to make "EMG" and "IMU" data of same length """

        x = np.arange(0, dattemp.shape[0])
        fit = scipy.interpolate.interp1d(x, dattemp, axis=0)
        dattemp = fit(np.linspace(0, dattemp.shape[0] - 1, datlength))
        return dattemp


class SubsamplingRecordings:
    def __init__(self):
        pass

    def subsample_data(self, datON, datOFF, detailsON, detailsOFF, modeltype, ratio, tasks):
        """in order to create a test and a validation dataset, this part just randomly samples a specific ratio of
        recordings and assigns them as test data; the output are two different datasets with all available data.
        also for multihead models data is splitted into task specific matrices
        TODO there are two conditionals on modeltype necessary. merge?"""

        if (modeltype == 'mh') | (modeltype == 'mc'):
            # data preprocessing for multihead and multichannel model
            datONout      = list()
            detailsONout  = list()
            datOFFout     = list()
            detailsOFFout = list()
            outONlen  = []
            outOFFlen = []
            cnt = 0
            for task in np.unique(detailsON.task):
              # split data into task specific matrices
                datONout.append(datON  [detailsON ['task'] == task, :, :])
                datOFFout.append(datOFF[detailsOFF['task'] == task, :, :])

                detailsONout.append( detailsON [detailsON ['task'] == task])
                detailsOFFout.append(detailsOFF[detailsOFF['task'] == task])

                outONlen.append( datONout [cnt].shape[0])
                outOFFlen.append(datOFFout[cnt].shape[0])

                cnt = cnt + 1

            minOFFlen = min(outOFFlen)
            minONlen  = min(outONlen)
            cnt = 0
            for task in tasks:
                # shave all task matrices to same length
                # TODO: this will always delete data of the last subject.
                #       it may be more sensible to randomize data first
                datONout [cnt] = datONout [cnt][0:minONlen , :, :]
                datOFFout[cnt] = datOFFout[cnt][0:minOFFlen, :, :]

                detailsONout [cnt] = detailsONout [cnt][0:minONlen]
                detailsOFFout[cnt] = detailsOFFout[cnt][0:minOFFlen]

                cnt = cnt +1
        else:
            minONlen  = datON.shape [0]
            minOFFlen = datOFF.shape[0]

      # create random indices for splitting data in training/testing subsamples
        trainingON_idx  = np.random.randint(minONlen , size=int(round(minONlen  * ratio)))
        trainingOFF_idx = np.random.randint(minOFFlen, size=int(round(minOFFlen * ratio)))

        testON_idx  = np.setdiff1d(np.arange(minONlen) , trainingON_idx )
        testOFF_idx = np.setdiff1d(np.arange(minOFFlen), trainingOFF_idx)

        if (modeltype == 'mh') | (modeltype == 'mc'):
            cnt = 0
            smplsONtrain  = list()
            smplsONtest   = list()
            smplsOFFtrain = list()
            smplsOFFtest  = list()
            for task in tasks:
                smplsONtrain.append( datONout [cnt][trainingON_idx  , :, :])
                smplsONtest.append(  datONout [cnt][testON_idx      , :, :])
                smplsOFFtrain.append(datOFFout[cnt][trainingOFF_idx , :, :])
                smplsOFFtest.append( datOFFout[cnt][testOFF_idx     , :, :])
                cnt = cnt + 1
        else:
            smplsONtrain,  smplsONtest  = datON [trainingON_idx , :, :], datON [testON_idx , :, :]
            smplsOFFtrain, smplsOFFtest = datOFF[trainingOFF_idx, :, :], datOFF[testOFF_idx, :, :]

        updrsONtrain  = detailsON.updrsON [trainingON_idx]
        updrsONtest   = detailsON.updrsON [testON_idx]
        updrsOFFtrain = detailsON.updrsOFF[trainingOFF_idx]
        updrsOFFtest  = detailsON.updrsOFF[testOFF_idx]

        return smplsONtrain, smplsONtest, smplsOFFtrain, smplsOFFtest, updrsONtrain, updrsONtest, updrsOFFtrain, updrsOFFtest

    def subsample_dataMOD(self, data, details, conditions, ratio=.8):
        """in order to create a test and a validation dataset, this part just randomly samples a specific ratio of
        recordings and assigns them as test data; the output are two different datasets with all available data.
        also for multihead models data is splitted into task specific matrices"""

        np.random.seed()
        rng = default_rng()
        details_test  = {k: [] for k in conditions}
        details_train = {k: [] for k in conditions}
        test_data = {k: [] for k in conditions}
        train_data = {k: [] for k in conditions}

        for cond in conditions:
            devices = list(data[cond].keys())

            if not devices:
                warnings.warn("No devices detected, please double-check!")
                return

            number_recordings = np.stack(data[cond][devices[0]]).shape[1]
            training_idx  = rng.choice(number_recordings, size=int(round(number_recordings  * ratio)), replace=False)
            test_idx  = np.setdiff1d(np.arange(number_recordings), training_idx)

            for dev in devices:
                # Extract training and test dataset, using random numbers
                data_train_temp = []
                [data_train_temp.append(data[cond][dev][x].iloc[training_idx,:]) for x in range(3)]

                data_test_temp = []
                [data_test_temp.append(data[cond][dev][x].iloc[test_idx,:]) for x in range(3)]

            test_data[cond] = data_test_temp
            train_data[cond] = data_train_temp
            details_train[cond] = details[cond].iloc[training_idx, :]
            details_test[cond] = details[cond].iloc[test_idx, :]

        return test_data, train_data, details_test, details_train


    def create_cat(self, X, Y, x, y, modeltype, outputtype):
        """this function establishes the categories for the data, i.e. whether it is an 'ON' or 'OFF' condition and
        concatenates all available recordings into a single matrix"""
        if (modeltype == 'mh') | (modeltype == 'mc'):
            cats = np.zeros(X[0].shape[0] + Y[0].shape[0])
            cats[0:X[1].shape[0]] = 1
            cats = to_categorical(cats)
            datAll = list()
            for i in range(0,len(X)):
                datAll.append(np.concatenate([X[i], Y[i]]))
        else:
            cats = np.zeros(X.shape[0] + Y.shape[0])
            cats[0:X.shape[0]] = 1
            cats = to_categorical(cats)
            datAll = np.concatenate([X, Y])

        if outputtype == 'reg':
            cats = np.concatenate([x, y])

        return datAll, cats

class Configuration:
    def __init__(self):
        pass

    @staticmethod
    def extract_options(CONFIGDATA):
        """at this point, all configurations stored in the file: config.yaml are extracted and returned"""
        import getpass

        scaling = CONFIGDATA['dataworks']['scaling']
        patlist = CONFIGDATA['dataworks']['folders'][getpass.getuser()]['patlist']

        conditions = CONFIGDATA['dataworks']['conds']
        ['ON', 'OFF'] if conditions == '' or conditions == 'all' else conditions

        tasks = CONFIGDATA['dataworks']['tasks']
        ['rst', 'hld', 'dd', 'tap'] if tasks == '' or tasks == 'all' else tasks

        devices = CONFIGDATA['dataworks']['dev']
        ['ACC', 'GYRO'] if devices == '' or devices == 'all' else devices

        return scaling, patlist, conditions, tasks, devices
