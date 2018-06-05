'''
Created on 2018年5月16日

@author: HW
'''
#在线导入数据集
from sklearn.datasets import load_iris
import matplotlib.pyplot as plot
import numpy as np
from matplotlib.pyplot import subplot
from mpl_toolkits.mplot3d import Axes3D


#逻辑回归的类
class logistic(object):
    def __init__(self):
        self.W = None
    
    def train(self,X,y,learn_rate = 0.01,num_iters = 5000):
        num_train,num_feature = X.shape
        
        #初始化权重
        self.W = np.random.randn(num_feature,1).reshape((-1,1))   #转换为列向量
        loss = []
        
        for i in range(num_iters):
            error,dW = self.compute_loss(X,y)
            self.W += -learn_rate*dW
            
            loss.append(error)
            if i%200 == 0:
                print('i = %d,error = %f'%(i,error))
        return loss
    
    def compute_loss(self,X,y):
        num_train = X.shape[0]   #数据集数量
        h = self.output(X)
        loss = -np.sum((y*np.log(h) + (1-y)*np.log((1-h))))
        loss = loss / num_train
        
        dW = X.T.dot((h-y)) / num_train
        return loss,dW
    
    def output(self,X):
        g = np.dot(X,self.W)
        return self.sigmod(g)
    
    def sigmod(self,X):
        return 1/(1+np.exp(-X))
    def predict(self,X_test):
        h = self.output(X_test)
        y_pred = np.where(h>0.5,1,0)
        return y_pred
    


iris = load_iris()
data = iris.data
target = iris.target

X = data[:,[0,1,2]]
y = target[:]

label = np.array(y)
index_0 = np.where(label == 0)
#
fig = plot.figure()
ax = fig.add_subplot(111,projection='3d')
index_1 = np.where(label == 1)
index_2 = np.where(label == 2)

ax.scatter(X[index_0,0],X[index_0,1],X[index_0,2],marker='x',color='b',label='0',s=15)
ax.scatter(X[index_1,0],X[index_1,1],X[index_1,2],marker='o',color='r',label='1',s=15)
ax.scatter(X[index_2,0],X[index_2,1],X[index_2,2],marker='v',color='y',label='2',s=15)


# subplot(1,2,1)
# plot.scatter(X[index_0,0],X[index_0,1],marker='x',color = 'b',label='0',s = 15)
# index_1 = np.where(label == 1)
# plot.scatter(X[index_1,0],X[index_1,1],marker='o',color = 'r',label='1', s = 15)
# index_2 = np.where(label == 2)
# plot.scatter(X[index_2,0],X[index_2,1],marker='v',color = 'y',label='2', s = 15)
# plot.xlabel('X1')
# plot.ylabel('X2')
# plot.legend(loc=2)

#测试并跟踪损失loss
# y = y.reshape((-1,1))
# one = np.ones([X.shape[0],1])
# X_train = np.hstack((one,X))  #合并成一个nx3的矩阵
# classify = logistic()
# loss = classify.train(X_train, y)
# print(classify.W)
# subplot(1,2,2)
# plot.plot(loss)
# plot.xlabel('Iteration number')
# plot.ylabel('Loss value')
plot.show()
