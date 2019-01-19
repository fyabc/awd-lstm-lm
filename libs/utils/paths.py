#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

__author__ = 'fyabc'

ProjectRoot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DataPath = os.path.join(ProjectRoot, 'data')
ModelPath = os.path.join(ProjectRoot, 'models')
ReservedDataPath = os.path.join(ProjectRoot, 'reserved_data')
CdfPath = os.path.join(ReservedDataPath, 'cdf.pkl')
