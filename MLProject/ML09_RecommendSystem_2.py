'''
Created on 2018年6月4日

@author: HW
'''

from matplotlib import pyplot
import scipy as sp
import numpy as np
from ML_GetData import readFile
import sys

#计算余弦距离
def getCosDist(user1, user2):
    sumx = 0.0
    sumy = 0.0
    sumxy = 0.0
    for k1 in user1:
        for k2 in user2:
            if k1[0] == k2[0]:
                sumx += k1[1]*k1[1]
                sumy += k2[1]*k2[1]
                sumxy += k1[1]*k2[1]
    if sumxy == 0.0:
        return 0
    demo = np.sqrt(sumx * sumy)
    return sumxy / demo

def getMoiveList(filename):
    with open()






if __name__ == '__main__':
    contents = readFile('data/ml-100k/u.data', '\t', 3, encoding='utf-8',DataStyle='int')
    contents = np.array(contents)
    
    moives = getMoiveList('u.item')
    
    print(contents)