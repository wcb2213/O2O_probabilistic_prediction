#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/8/15


import numpy as np
import pandas as pd


dftest = pd.read_csv('data/ccf_offline_stage1_test_revised.csv')
dfoff = pd.read_csv('data/ccf_offline_stage1_train.csv')
## dfon 无 distance

distance1 = dftest['Distance']
distance2 = dfoff['Distance']
print("distance1:", distance1.mean(), pd.Series(distance1).value_counts(normalize=True))
# mean:2.3280400882098133     p: 0.0  0.431864
print("distance2:", distance2.mean(), pd.Series(distance2).value_counts(normalize=True))
# mean:2.3616355576903367     p: 0.0  0.500988