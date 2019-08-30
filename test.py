#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/8/14

import numpy as np
import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', None)

dftest = pd.read_csv('processedData/dftest.csv')
dfoff = pd.read_csv('processedData/dfoff.csv')
dfon = pd.read_csv('processedData/dfon.csv')

print(dfoff.shape)
print(dfon.shape)
df = pd.concat([dfoff,dfon])
print(df.shape)

# 2 缺失值
print(dftest.isnull().any())
# 无
print(dfoff.isnull().any())
# 无
print(dfon.isnull().any())
# discount_rate     True  因为 fixed的问题
# discount_jian     True

print(dfon['discount_jian'].unique().shape)
print(np.sort(dfon['discount_jian'].unique()))