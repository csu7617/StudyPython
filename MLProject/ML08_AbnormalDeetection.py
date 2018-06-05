'''
Created on 2018年5月31日

@author: HW
'''

import numpy as np
import pandas as pd
from ML_GetData import GetData



def getU(X_train):
    u = []
    row = X_train.shape[0]
    line = X_train.shape[1]
    for i in range(X_train.shape[1]):
        u.append(X_train[:,i].sum()/row)
    u = np.array(u)
    return u

def getSigma2(X_train,u):
    sigma2 = []
    row = X_train.shape[0]
    line = X_train.shape[1]
    if X_train.shape[1] == 1:
        std2 = np.mean(np.square(X_train-u),axis=0)
    else:
        std2 = np.dot(np.transpose(X_train-u),(X_train-u))/X_train.shape[0]
        std2 += np.eye(X_train.shape[1])
    
#     for i in range(line):
#         sum = 0
#         for j in range(row):
#             sum += (X_train[j][i]-u[i])*(X_train[j][i] - u[i])
#         sigma2.append(sum/row)
#     sigma2 = np.array(sigma2)
    return std2

def AbnormalDetection(u,sigma2,xi):
    px = 1
    if xi.shape[0] == 1:
        p = ((1/np.sqrt(2*np.pi*sigma2))*np.exp(-(xi - u)*(xi-u)/(2*sigma2)))
        px = p
    else:
        p = 1/(np.power(2*np.pi,float(xi.shape[0])/2)*np.sqrt(np.linalg.det(sigma2)))*np.exp((-1/2)*np.dot(np.dot(xi-u,np.linalg.pinv(sigma2)),np.transpose(xi-u)))
        px = p
#     p = ((1/np.sqrt(2*np.pi*sigma2))*np.exp(-(xi - u)*(xi-u)/(2*sigma2)))
#     px = 1
#     for i in range(len(p)):
#         px *= p[i]
    return px


if __name__ == '__main__':
    X_train,X_test,y_train,y_test = GetData('iris', 0)
    x = [[1,2,3,4],[2,3,4,5],[3,4,5,6]]
#     for i in range(X_test.shape[0]):
#     print(np.array(x), np.array([1,2,3,4]))
    u = getU(np.array(x))
    sigma2 = getSigma2(np.array(x), u)
    print(AbnormalDetection(u,sigma2, np.array([1,2,3,4])),'--------\n')
