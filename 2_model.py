#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/8/27


import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from sklearn.linear_model import SGDClassifier, LogisticRegression


def train():
    #### load data
    print("-----load data------")
    dfoff = pd.read_csv('processedData/dfoff.csv')
    dfon = pd.read_csv('processedData/dfon.csv')
    dfon = dfon[pd.notnull(dfon['discount_rate'])].copy()

    dftrain = pd.concat([dfoff, dfon])

    #### feature and train
    df = pd.get_dummies(dftrain, columns=['discount_type','weekday','weekday_type'])
    label = df['label']
    df.drop(['label'], axis=1, inplace=True)

    print("----train-----")
    model = SGDClassifier(#lambda:
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        max_iter=100,
        shuffle=True,
        alpha = 0.01,
        l1_ratio = 0.01,
        n_jobs=1,
        class_weight=None
    )
    model.fit(df, label)

    #### save model
    print("---save model---")
    with open('model/4_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def save_result():
    #### load data
    dftest = pd.read_csv('processedData/dftest.csv')
    dftest = pd.get_dummies(dftest, columns=['discount_type', 'weekday', 'weekday_type'])

    with open('model/4_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # test prediction for submission
    y_test_pred = model.predict_proba(dftest)

    # save the results
    original_test = pd.read_csv('data/ccf_offline_stage1_test_revised.csv')
    original_test = original_test[['User_id','Coupon_id','Date_received']].copy()
    original_test['label'] = y_test_pred[:,1]
    original_test.to_csv('result/submit4.csv', index=False, header=False)
    original_test.head()


if __name__ == '__main__':
    # train()
    save_result()