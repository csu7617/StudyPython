'''
Created on 2018年5月22日

@author: HW
'''
from numpy import rate

'''
机器学习获取数据的方法
'''

from sklearn import datasets
from sklearn.model_selection import train_test_split
import os
import numpy as np
import gzip

def readFile(filename,split,k,encoding='utf-8',DataStyle='str'):
    contents = []
    with open(filename,mode='r+',encoding=encoding) as f:
        for line in f:
            line = line.strip()
            rate = line.split(split)
            c = []
            for i in range(k):
                if DataStyle == 'str':
                    c.append(rate[i])
                elif DataStyle == 'int':
                    c.append(int(rate[i]))
                elif DataStyle == 'float':
                    c.append(float(rate[i]))
            contents.append(c)
    return np.array(contents)

def GetIrisData():
    data = datasets.load_iris()
    X = data.data  #获取特征数据
    y = data.target #标签矩阵
#     print(y)
    y = y.reshape(-1,1)
    X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.28,random_state = 5)  #分割训练集
    
    return X_train,X_test,y_train,y_test

def GetIrisData_1():
    data = datasets.load_iris()
    X = data.data  #获取特征数据
    y_1 = data.target #标签矩阵
    y_1 = y_1.reshape(-1,1)
    '''
            将标签列分成n列，n为样本类
    '''
    y = []
    length = list(set(y_1.reshape(-1).tolist()))
    for i in range(y_1.shape[0]):
        temp = []
        for j in range(length):
            if y_1[i][0] == j:
                temp.append(1)
            else:
                temp.append(0)
        y.append(temp)
    X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.28,random_state = 8)  #分割训练集
    return X_train,X_test,y_train,y_test

def GetIrisData_2():
    data = datasets.load_iris()
    X = data.data  #获取特征数据    
    y = data.target  #获取标签
    
    X = np.c_[X,y]
    return X

def GetForestData():
    result = []
    input_answer = []
    k = 0
    with gzip.open('data/covtype/covtype.data.gz','rt') as pf:
        for line in pf:
            if k>1000:
                break;
            k += 1
            line = line.strip()
            result.append([int(feature) for feature in line.split(",")])
    data = np.array(result)
    X = data[:,0:data.shape[1]-1]
    #归一化处理
#     X = (X - np.mean(X,axis = 0))/np.std(X,axis=0)
    y = data[:,data.shape[1]-1:data.shape[1]]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=8)
    return X_train,X_test,y_train,y_test

'''
'''
def GetForestData_1():
    result = []
    input_answer = []
    with gzip.open('data/covtype/covtype.data.gz', 'rt') as pf:
        for line in pf:
            line = line.strip()
            result.append([int(feature) for feature in line.split(',')])
    data = np.array(result)
    X = data[:,0:data.shape[1]-1]
    #归一化处理
    X = (X - np.mean(X,axis = 0))/np.std(X,axis=0)
    y_1 = data[:,data.shape[1]-1:data.shape[1]]
    '''
            将标签列分成n列，n为样本类
    '''
    y = []
    length = list(set(y_1.reshape(-1).tolist()))
#     print(y_1,y_1.shape[0])
    for i in range(y_1.shape[0]):
        temp = []
        for j in length:
            if y_1[i][0] == j:
                temp.append(1)
            else:
                temp.append(0)
        y.append(temp)
#     print(y,y_1)
    y = np.array(y)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=8)
    return X_train,X_test,y_train,y_test
'''
str: 表明调用的是哪个数据
flag: 表明该数据的标签是否由1列分成n列
'''
def GetData(str,flag):
    if str == 'iris' or str == 'Iris':
        if flag == 0:
            return GetIrisData()
        elif flag == 1:
            return GetIrisData_1()
        elif flag == 2:
            return GetIrisData_2()
    if str == 'forest' or str == 'Forest':
        if flag == 0:
            return GetForestData()
        else:
            return GetForestData_1()
    
if __name__ == '__main__':
    X_train,X_test,y_train,y_test = GetData('iris',0)
#     print(y_train,y_test)
    
#     X = GetData('forest', 0)
#     print(X)

    