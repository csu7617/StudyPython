'''
Created on 2018年5月16日

@author: HW
'''




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#计算代价
def computerCost(X,y,theta):
    inner = np.power(((X*theta.T)-y),2)
    return np.sum(inner) / (2*len(X))

#利用向量计算梯度
def gradientDescent(X,y,theta,alpha,epoch):
    temp = np.matrix(np.zeros(theta.shape))   #初始化一个θ的临时矩阵  矩阵.shape返回一个包含各维度数量的元组
    cost = np.zeros(epoch)                    #初始化一个代价数组,每一次循环存储一个代价
    m = X.shape[0]                            #样本数量
    
    for i in range(epoch):
        #利用向量化一步求解
        temp = theta - (alpha/m)*(X*theta.T-y).T*X
        theta = temp
        cost[i] = computerCost(X, y, theta)
    return theta,cost


path = "data//ex1data1.txt"
#names添加列名，header用指定的行为作为标题，若无原标题则设为None
data = pd.read_csv(path,header=None,names=['Population','Profit'])

#我们在训练集中添加一列，以便我们可以使用向量化的结局方案来计算大家和梯度
data.insert(0,'Ones',1)           #即(x,1) * (a,b)的转置
cols = data.shape[1]  #列数    shape是矩阵的属性,shape[0]是矩阵的行数
X = data.iloc[:,0:cols-1]    #,前取行， ‘，’后取列，
y = data.iloc[:,cols-1:cols]

X = np.matrix(X.values)   #将X转换为矩阵
y = np.matrix(y.values)   #
theta = np.matrix([0,0])
#指定学习率和循环次数
alpha = 0.01
epoch = 1000
#开始梯度下降
final_theta,cost = gradientDescent(X, y, theta, alpha, epoch)

#绘制线性模型以及数据，直观地看出它的拟合
x = np.linspace(data.Population.min(),data.Population.max(),100)   #横坐标
f = final_theta[0,0] +(final_theta[0,1]*x)                         #纵坐标 ，利润 b+ax

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x,f,'r',label='Prediction')                               #拟合出来的线
ax.scatter(data['Population'],data.Profit,label='Traning Data')
ax.legend(loc = 2)                                        #2表示在左上角
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

 
