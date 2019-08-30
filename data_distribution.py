#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/8/24


import numpy as np
import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', None)

dftest = pd.read_csv('data/ccf_offline_stage1_test_revised.csv')
#    User_id  Merchant_id  Coupon_id Discount_rate  Distance  Date_received
# 0  4129537          450       9983          30:5       1.0       20160712
# 1  6949378         1300       3429          30:5       NaN       20160706
# 2  2166529         7113       6928        200:20       5.0       20160727
# 3  2166529         7113       1808        100:10       5.0       20160727
# 4  6172162         7605       6500          30:1       2.0       20160708
# (113640, 6) ['User_id' 'Merchant_id' 'Coupon_id' 'Discount_rate' 'Distance' 'Date_received']

print(dftest['User_id'].unique().shape) #(76309,)
print(dftest['Merchant_id'].unique().shape) #(1559,)
print(dftest['Coupon_id'].unique().shape) #(2050,)
print(dftest['Discount_rate'].unique()) #(42,)
#['30:5' '200:20' '100:10' '30:1' '50:10' '10:1' '150:30' '20:1' '20:5'
 # '100:20' '300:50' '50:5' '0.9' '50:1' '10:5' '100:5' '150:5' '0.8'
 # '300:30' '5:1' '0.95' '150:20' '200:50' '50:20' '30:10' '100:30' '100:50'
 # '200:30' '150:50' '30:20' '200:10' '500:30' '20:10' '0.5' '150:10'
 # '200:5' '300:20' '100:1' '200:100' '0.85' '0.6' '0.7']
print(np.sort(dftest['Distance'].unique())) #[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. nan]         注意 存在缺失值
print(dftest['Date_received'].unique().shape) #(31,)
print(np.sort(dftest['Date_received'].unique())) #20160701-20160731


dfoff = pd.read_csv('data/ccf_offline_stage1_train.csv')
#    User_id  Merchant_id  Coupon_id Discount_rate  Distance  Date_received        Date
# 0  1439408         2632        NaN           NaN         0            NaN  20160217.0
# 1  1439408         4663    11002.0        150:20         1     20160528.0         NaN
# 2  1439408         2632     8591.0          20:1         0     20160217.0         NaN
# 3  1439408         2632     1078.0          20:1         0     20160319.0         NaN
# 4  1439408         2632     8591.0          20:1         0     20160613.0         NaN
# (1754884, 7) ['User_id' 'Merchant_id' 'Coupon_id' 'Discount_rate' 'Distance' 'Date_received' 'Date']

print(dfoff['User_id'].unique().shape) #(539438,)
print(dfoff['Merchant_id'].unique().shape) #(8415,)
print(dfoff['Coupon_id'].unique().shape) #(9739,)                                                      注意 存在缺失值
print(dfoff['Discount_rate'].unique()) #(46,)                                                          注意 存在缺失值
# [nan '150:20' '20:1' '200:20' '30:5' '50:10' '10:5' '100:10' '200:30'
#  '20:5' '30:10' '50:5' '150:10' '100:30' '200:50' '100:50' '300:30'
#  '50:20' '0.9' '10:1' '30:1' '0.95' '100:5' '5:1' '100:20' '0.8' '50:1'
#  '200:10' '300:20' '100:1' '150:30' '300:50' '20:10' '0.85' '0.6' '150:50'
#  '0.75' '0.5' '200:5' '0.7' '30:20' '300:10' '0.2' '50:30' '200:100'
#  '150:5']
print(np.sort(dfoff['Distance'].unique())) #[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. nan]          注意 存在缺失值
print(dfoff['Date_received'].unique().shape) #(168,)                                                   注意 存在缺失值
print(np.sort(dfoff['Date_received'].unique())) #20160101-20160615 + nan
print(dfoff['Date'].unique().shape) #(183,)                                                             注意 存在缺失值
print(np.sort(dfoff['Date'].unique())) #20160101-20160630 + nan



dfon = pd.read_csv('data/ccf_online_stage1_train.csv')
# (11429826, 7) ['User_id' 'Merchant_id' 'Action' 'Coupon_id' 'Discount_rate' 'Date_received' 'Date']
#     User_id  Merchant_id  Action    Coupon_id Discount_rate  Date_received        Date
# 0  13740231        18907       2  100017492.0        500:50     20160513.0         NaN
# 1  13740231        34805       1          NaN           NaN            NaN  20160321.0
# 2  14336199        18907       0          NaN           NaN            NaN  20160618.0
# 3  14336199        18907       0          NaN           NaN            NaN  20160618.0
# 4  14336199        18907       0          NaN           NaN            NaN  20160618.0
#
print(dfon['User_id'].unique().shape) #(762858,)
print(dfon['Merchant_id'].unique().shape) #(7999,)
print(dfon['Action'].unique()) #[2 1 0]
print(dfon['Coupon_id'].unique().shape) #(27748,)                                                      注意 存在缺失值
print(dfon['Discount_rate'].unique().shape) #(65,)                                                     注意 存在缺失值
# ['500:50' nan '150:50' '50:5' '30:1' '300:50' '300:30' '800:50' '1000:100'
#  '10:5' '200:30' '20:10' '200:20' '150:10' '300:10' '150:1' '200:10'
#  'fixed' '5:1' '50:10' '100:10' '100:5' '150:20' '200:50' '300:5' '500:10'
#  '300:100' '50:30' '50:1' '100:1' '30:20' '10:1' '30:5' '100:50' '150:5'
#  '150:30' '300:20' '500:30' '800:100' '50:20' '500:100' '500:20' '1000:20'
#  '1000:50' '100:30' '30:10' '200:5' '200:100' '150:100' '200:1' '20:5'
#  '1000:10' '20:1' '100:20' '300:1' '500:5' '800:20' '800:30' '1000:300'
#  '1000:30' '800:10' '1000:500' '500:300' '1000:5' '800:500']
print(dfon['Date_received'].unique().shape) #(168,)                                                   注意 存在缺失值
print(np.sort(dfon['Date_received'].unique())) #20160101-20160615 + nan
print(dfon['Date'].unique().shape) #(183,)                                                             注意 存在缺失值
print(np.sort(dfon['Date'].unique())) #20160101-20160630 + nan

## 1 shape
# print(dftest.shape,dftest.columns.values)

## 2 缺失值
# print(dftest.isnull().any())
# Distance          True
# test中所有其他的distance的值的平均值

# print(dfoff.isnull().any())
# Coupon_id         True
# Discount_rate     True
# Distance          True
# Date_received     True
# Date              True

# print(dfon.isnull().any())
# Coupon_id         True
# Discount_rate     True
# Date_received     True
# Date              True

