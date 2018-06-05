'''
Created on 2018年5月19日

@author: HW
'''
from sklearn.cross_validation import cross_val_score

'''
决策树算法在谷歌的莺尾花数据集上实现分类，并在二维图上显示
与随机森林进行比较
'''


import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import tree
import numpy as np
import datetime


def DecisonTree(X_train,y_train,X_test,y_test):  
    #训练模型，限制树的最大深度4
    clf = tree.DecisionTreeClassifier(criterion='gini',random_state=8)   #决策树
    start_time = datetime.datetime.now()
    print('--------决策树---------')
    clf.fit(X_train,y_train)
    predict = clf.predict(X_test)
    #评估得出的分数
    print('预测准确率 : %0.2f'%np.mean(predict == y_test.reshape(-1)))
    cvs = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv=10)
    print('训练集平均准确率: %0.2f(+/- %0.2f)'%(cvs.mean(),cvs.std() * 2))
    end_time = datetime.datetime.now()
    time_spend = end_time - start_time
    print('花费时间： %0.2f'%time_spend.total_seconds())  
    #画图
    # x_min,x_max = X_train[:,0].min(),X_train[:,0].max()
    # y_min,y_max = X_train[:,2].min(),X_train[:,2].max()
    # xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    # f = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    # f = f.reshape(xx.shape)
    # plot.xlabel("X label",color='r')
    # plot.ylabel("Y label",color='b')
    # # ax.plot_surface( xx,yy, zz,rstride=1,cstride=1)
    # plot.contour(xx,yy,f,colors='black')
    # plot.scatter(X_train[:,0],X_train[:,1],c = y,alpha=0.8)
    # plot.show()

if __name__ == '__main__':
    DecisonTree()







