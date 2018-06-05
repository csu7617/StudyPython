'''
Created on 2018年5月24日

@author: HW
'''


from ML_GetData import GetData
from numpy import *
import numpy as np




#计算两点间的距离
def getDistance(x,y):
    sum = 0.0
    for i in range(4):
        sum = sum + (x[i]-y[i])*(x[i]-y[i])
    return np.sqrt(sum)

#根据执行，决定当前元组属于哪个簇
def clusterOfTuple(k_center,X_i):
    min = getDistance(k_center[0],X_i)
    
    label = 0  #标识 属于哪一类
    for i in range(1,3):
        temp = getDistance(k_center[i], X_i)
        if temp < min:
            min = temp
            label = i
    return label

#获得给定簇集的误差平方和
def getVal(k,k_cluster,k_center):
    var = 0
    for i in range(k):
        cluster = k_cluster[i]
        temp = 0
        for j in range(1,len(cluster)):
            temp = temp + getDistance(cluster[j], k_center[i])
        var += temp/(len(cluster)-1)
    return var

def getMean(cluster):
    cluster = cluster[1:-1]
    cluster = np.array(cluster)
    
    num = cluster.shape[0]
    t = []
    for i in range(4):
        t.append(0.0)
    for i in range(num):
        for j in range(4):
            t[j] += cluster[i][j]
    for i in range(4):
        t[i] /= num
        
    #去掉最后一列标签列
    print(t)
    return t

def KMeans(X,k,feature_size):
    '''
    X: 带标签数据
    k: 类别
    feature_size: 特征量
    '''
    k_cluster = []
    k_center = []
    #初始化三个簇的
    for i in range(k):
        k_cluster.append([i])
    #一开始随机选取k条记录的值作为k个簇的质心
    n = [5,88,121]
    for i in range(k):
        temp = X[n[i]]
        k_center.append(temp[0:len(temp)-1])
    k_center = np.array(k_center).astype(np.float32)
    #根据默认的质心给簇赋值
    for i in range(X.shape[0]):
        label = clusterOfTuple(k_center, X[i])
        k_cluster[label].append(X[i])
    
    oldVal = -1
    newVal = getVal(k,k_cluster,k_center)
    print('初始的整体误差平方和：%f'%newVal)
    #指定迭代次数
    t = 0
    while(t<100):
        print('第',t,'此迭代开始：')
        t = t + 1
        #更新每个簇的中心点
        for i in range(k):
            k_center[i] = getMean(k_cluster[i])
        oldVal = newVal
        newVal = getVal(k, k_cluster, k_center)
        #清空每个簇
        k_cluster = []
        for i in range(k):
            k_cluster.append([0])
        #根据新的执行获得新的簇
        for i in range(X.shape[0]):
            label = clusterOfTuple(k_center, X[i])
            k_cluster[label].append(X[i])
        print('前一次的误差平方和的平均：')
        print('迭代后的整体误差平方和为：%f'%newVal)
    #打印分类结果
    for i in range(k):
        print('第%d个簇: '%i)
        t = k_cluster[i]
        for j in range(len(t)):
            print(t[j])
        print('-------------')
    

if __name__ == '__main__':
    fname = []
    dimNum = 4
    dataNum = 150
    
    X = GetData('iris',2)
    #开始聚类
    KMeans(X,3,4)
    