'''
Created on 2018年5月22日

@author: HW
'''
from sklearn.grid_search import GridSearchCV


'''
使用随机森林预测森林植被
'''

import datetime
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np


#自行选择最优组合
def RF_PredictForestAreas_1(X_train,y_train,X_test,y_test):
    starttime = datetime.datetime.now()
    #给最小叶子结点数和树的个数赋值，形成一个数组，看哪个区间随机森林的效果最优
    min_leaf_nodes = []
    n_estimators = []
    y = 0
    for i in range(1,20): 
        y = i * (i+5)
        min_leaf_nodes.append(1 + i*i)
        n_estimators.append(y)   
    for i in range(1,10):
        n_estimators.append(y + 50*i)        
    
#     for leaf in min_leaf_nodes:
    for estimator in n_estimators:
        #训练模型
        #n_estimators: 表示树的数量
        rfc = RandomForestClassifier(criterion='gini',n_estimators=estimator,min_samples_leaf=3,bootstrap=True,random_state=8)
        #拟合训练集
        rfc.fit(X_train,y_train)  
        predict = rfc.predict(X_test)      
        print('------随机森林预测准确率--------')
        print('最小叶子结点数： %d 树的个数：%d'%(3,estimator))
        print('训练集准确率：%0.2f'%np.mean(y_train == rfc.predict(X_train)))
        print('测试集准确率: %0.2f'%np.mean(predict == y_test.reshape(-1)))
        #scores = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv = 10)
        #print('训练集准确率：%0.2f(+/- %0.2f)'%(scores.mean(),scores.std()*2))
        endtime = datetime.datetime.now()
        spendtime = endtime - starttime
        print('花费时间：%0.2f'%spendtime.total_seconds())

    
    
#默认方式解决问题
def RF_PredictForestAreas_2(X_train,y_train,X_test,y_test):
    starttime = datetime.datetime.now()
    #训练模型
    rfc = RandomForestClassifier(n_estimators=50,criterion='gini',bootstrap=True,random_state=8)
    rfc.fit(X_train,y_train)
    
    predict = rfc.predict(X_test)
    print('------------随机森林预测准确率----------')
    print('测试集准确率: %0.2f'%np.mean(predict==y_test.reshape(-1)))
    scores = cross_val_score(rfc,X_train,y_train,scoring='accuracy',cv=10)
    print('训练集准确率: %0.2f(+/- %0.2f)'%(scores.mean(),scores.std()*2))
    endtime = datetime.datetime.now()
    spendtime = endtime-starttime
    print('花费时间: %0.2f'%spendtime.total_seconds())
    
    
    
    
    
    
    
