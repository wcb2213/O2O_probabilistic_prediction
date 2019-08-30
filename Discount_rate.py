#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/8/15


import pandas as pd


dftest = pd.read_csv('data/ccf_offline_stage1_test_revised.csv')
dfoff = pd.read_csv('data/ccf_offline_stage1_train.csv')
dfon = pd.read_csv('data/ccf_online_stage1_train.csv')

# a1 = dftest['Discount_rate'].unique() #(42,)
# a2 = dfoff['Discount_rate'].unique() #(46,)
# a3 = dfon['Discount_rate'].unique() #(65,)

# res1 = []
# for i in a1:
#     if i not in a2:
#         res1.append(i)
# print(res1) # ['500:30']
# res2 = []
# for i in a2:
#     if i not in a1:
#         res2.append(i)
# print(res2) # [nan, '0.75', '300:10', '0.2', '50:30']
# res3 = []
# for i in a3:
#     if i not in a1 or i not in a2:
#         res3.append(i)
# print(res3) # ['500:50', nan, '800:50', '1000:100', '300:10', '150:1', 'fixed', '300:5', '500:10', '300:100', '50:30', '500:30', '800:100', '500:100', '500:20', '1000:20', '1000:50', '150:100', '200:1', '1000:10', '300:1', '500:5', '800:20', '800:30', '1000:300', '1000:30', '800:10', '1000:500', '500:300', '1000:5', '800:500']

max_number1, max_number2, max_number3 = [], [], []
min_number1, min_number2, min_number3 = [], [], []
for i in dftest['Discount_rate']:
    if ':' in str(i):
        i = str(i).split(':')
        max_number1.append(int(i[0]))
        min_number1.append(int(i[1]))
print("max_number:", pd.Series(max_number1).value_counts(normalize=True)) #30     0.574149
print("min_number:", pd.Series(min_number1).value_counts(normalize=True)) #5      0.633479
for i in dfoff['Discount_rate']:
    if ':' in str(i):
        i = str(i).split(':')
        max_number2.append(int(i[0]))
        min_number2.append(int(i[1]))
print("max_number:", pd.Series(max_number2).value_counts(normalize=True)) #30     0.295176
print("min_number:", pd.Series(min_number2).value_counts(normalize=True)) #5      0.431511
for i in dfon['Discount_rate']:
    if ':' in str(i):
        i = str(i).split(':')
        max_number3.append(int(i[0]))
        min_number3.append(int(i[1]))
print("max_number:", pd.Series(max_number3).value_counts(normalize=True)) #50      0.202506
print("min_number:", pd.Series(min_number3).value_counts(normalize=True)) #10     0.249610