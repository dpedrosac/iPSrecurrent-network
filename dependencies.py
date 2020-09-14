#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import getpass

ROOTDIR = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(ROOTDIR, 'config.yaml'), 'r') as f:
    CONFIGDATA = yaml.load(f.read(), Loader=yaml.FullLoader)

FILEDIR = CONFIGDATA[0]['dataworks']['folders'][getpass.getuser()]['datpath']
GITHUB = 'https://github.com/dpedrosac/iPScnn'