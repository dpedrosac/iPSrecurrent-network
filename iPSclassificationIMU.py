#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import preprocess.processing as preparedata
import preprocess.HelperFunctions as helpers
import warnings
import itertools

import numpy as np
import pandas as pds
import yaml

from dependencies import ROOTDIR, FILEDIR ,CONFIGDATA

with open(os.path.join(ROOTDIR, 'config.yaml'), 'r') as f:
    CONFIGDATA = yaml.load(f.read(), Loader=yaml.FullLoader)
CONFIGDATA = CONFIGDATA[0]
scaling, patlist, conditions, tasks, devices = preparedata.Configuration.extract_options(CONFIGDATA)
datobj = preparedata.DataProcessing()
output = helpers.Output()

# General functions ans settings to run the scripts
device_list = ['ACC', 'GYRO'] # list of devices to be loaded/extracted/processing
channel_list = ['x', 'y', 'z']
combination_list = list(itertools.product(device_list, channel_list))
subject_list, index_list, metadata = preparedata.DataProcessing.generate_subjlist(filename_patlist=patlist)

# Start extractin or lorading data
if not os.path.isfile(os.path.join(ROOTDIR +'/data/IMU/dataON_'+ ''.join(combination_list[0])  + '.csv')):
    warnings.warn("\tData not available, starting to extract data for all subjects.")
    data, details = datobj.concatenate_recordingsMOD(subject_list, conditions, devices, tasks, metadata)

    for cond in conditions:
        for (i, j) in combination_list:
            filename = os.path.join(ROOTDIR, 'data/IMU/data' + cond + '_' + ''.join(i + j) + '.csv')
            data2save = data[cond][j][details[cond].device==i]
            details2save = details[cond][details[cond].device==i]
            np.savetxt(filename, data2save, delimiter=';')
            details2save.to_csv(os.path.join(ROOTDIR + '/data/IMU/details' + cond + '_' + i +  '.csv'),
                                mode='w', header=True)
    warnings.warn("Transformation done. Please restart the application to make things work!")
else:
    print("\t...loading already processed data to workspace")
    data = {k: [] for k in conditions}
    details = {k: [] for k in conditions}
    for cond in conditions:
        data[cond] = {k: [] for k in device_list}
        details[cond] = {k: [] for k in device_list}
        for (i, j) in combination_list:
            filename_data = os.path.join(ROOTDIR, 'data/IMU/data' + cond + '_' + ''.join(i + j) + '.csv')
            filename_details = os.path.join(ROOTDIR, 'data/IMU/details' + cond + '_' + i + '.csv')
            data[cond][i].append(pds.read_csv(filename_data, sep=';'))
            details[cond]= pds.read_csv(filename_details)

# Categorise data to prepare model estimation
catobj = preparedata.SubsamplingRecordings()
print("\t Subsampling data to train and test datasets")
test_data, train_data, details_test, details_train = catobj.subsample_dataMOD(data, details, conditions, ratio=.8)

