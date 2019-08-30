#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/8/14


import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from sklearn.linear_model import SGDClassifier, LogisticRegression


def processData(df):
    # 1. 将满xx减yy类型(`xx:yy`)的券变成折扣率 : `1 - yy/xx`，同时建立折扣券相关的特征 `discount_rate, discount_man, discount_jian, discount_type`
    # 2. 将距离 `str` 转为 `int`
    # convert Discount_rate and Distance
    def getDiscountType(row):
        if pd.isnull(row):
            return np.nan
        elif ':' in row:
            return 1
        else:
            return 0
    def convertRate(row):
        """Convert discount to rate"""
        if pd.isnull(row):
            return 1.0
        elif ':' in str(row):
            rows = row.split(':')
            return 1.0 - float(rows[1]) / float(rows[0])
        elif str(row)!='fixed':
            return float(row)
    def getDiscountMan(row):
        if ':' in str(row):
            rows = row.split(':')
            return int(rows[0])
        else:
            return 30
    def getDiscountJian(row):
        if ':' in str(row):
            rows = row.split(':')
            return float(rows[1])
        elif pd.notnull(row) and str(row)!='fixed':
            return 30 * float(row)
    # convert discunt_rate
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    return df

def getWeekday(row):
    if row == 'nan':
        return np.nan
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1

def label(row):
    if pd.isnull(row['Date_received']):
        return -1
    if pd.notnull(row['Date']):
        td = pd.to_datetime(row['Date'], format='%Y%m%d') -  pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0

def featureExtract():
    #### load data
    dftest = pd.read_csv('data/ccf_offline_stage1_test_revised.csv')
    dfoff = pd.read_csv('data/ccf_offline_stage1_train.csv')
    dfon = pd.read_csv('data/ccf_online_stage1_train.csv')
    print('data read end.')

    #### data process
    dftest = processData(dftest)
    dftest['distance'] = dftest['Distance'].fillna(2.3280).astype(float)
    dfoff = processData(dfoff)
    dfoff['distance'] = dfoff['Distance'].fillna(2.3616).astype(float)
    dfon = processData(dfon) ## dfon中的 discount_rate 含有 fixed
    dfon['distance'] = 0

    dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday) # [nan  6.  3.  1.  5.  4.  7.  2.]
    dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday) # 注意含有 nan
    dfon['weekday'] = dfon['Date_received'].astype(str).apply(getWeekday)

    # weekday_type :  周六和周日为1，其他为0
    dftest['weekday_type'] = dftest['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )
    dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )
    dfon['weekday_type'] = dfon['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )
    print('processData end.')

    #### label
    dfoff['label'] = dfoff.apply(label, axis = 1) # axis = 1 处理每一行
    dfon['label'] = dfon.apply(label, axis=1)
    print("label process end")

    dftest = dftest[
        ['discount_rate', 'discount_man', 'discount_jian', 'discount_type', 'distance', 'weekday', 'weekday_type']]
    dfoff = dfoff[
        ['discount_rate', 'discount_man', 'discount_jian', 'discount_type', 'distance', 'weekday', 'weekday_type',
         'label']]
    dfon = dfon[
        ['discount_rate', 'discount_man', 'discount_jian', 'discount_type', 'distance', 'weekday', 'weekday_type',
         'label']]
    dfoff = dfoff[dfoff['label'] != -1].copy()
    dfon = dfon[dfon['label'] != -1].copy()
    dftest.to_csv('processedData/dftest.csv', index=False)
    dfoff.to_csv('processedData/dfoff.csv', index=False)
    dfon.to_csv('processedData/dfon.csv', index=False)

featureExtract()