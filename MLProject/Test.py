'''
Created on 2018年5月14日

@author: Administrator
'''

import tensorflow as tf;  
import numpy as np;  
from sklearn import datasets
# from numpy import *
import random
import pandas as pd
import cx_Oracle


a = [[1,2,3],[2,3,4],[3,4,5]]
a= np.array(a)
u = []
for i in range(3):
    u.append(a[:,i].sum()/3)
u = np.array(u)



print(a-u)

































# df = pd.read_csv('C:/Users/HW/Desktop/FollowCar/tsdata180605.csv',header=None,sep=',')
# print(df.head())
# print(df.tail())
# contents = []
# index = 0
# contents.append([0,'JGSJ','HPHM','LKMC','FLAG'])
# names = ['A5NE51.csv','AN689M.csv','AR8R87.csv','ACX845.csv','ALN393.csv']
# for i in range(5):    
#     with open('C:/Users/HW/Desktop/DATA/' + names[i],mode='r+',encoding='utf-8') as f:
#         k = 0
# 
#         for line in f:
# #             if k > 100:
# #                 break
#             k += 1
#             line = line.strip()
#             temp = line.split('\t')
#             content = []
#             content.append(index)
#             for i in range(len(temp)):
#                 content.append(temp[i])
#             content.append("0")
# #             print(content)
#             if content[1] != '未识别':
#                 contents.append(content)
#             index +=1
# 
# filename = 'C:/Users/HW/Desktop/DATA/ts.txt'
# with open(filename,mode='w+',encoding='utf-8') as f:
#     for i in range(len(contents)):
#         temp = contents[i]
#         print(temp)
#         s = str(temp[0]) + ',' + temp[1] + ',' + temp[2] + ',' + temp[3] +',' + temp[4] + '\n'
#         f.write(s)

