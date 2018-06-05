'''
Created on 2018年5月30日

@author: HW
'''
from sklearn.cluster import KMeans
from ML_GetData import GetData
import numpy as np




if __name__ == '__main__':
    X = GetData('Iris',2)
    y = X[:,X.shape[1]-1:X.shape[1]]
    X = X[:,0:X.shape[1]-1]

    kmeans = KMeans(n_clusters=3,init='k-means++',n_jobs=-1,algorithm='auto').fit(X)
    y = y.reshape(-1)
#     print(kmeans.labels_,y)
    print(np.mean(kmeans.labels_==y))