'''
Created on 2018年5月22日

@author: HW
'''

'''
使用随机森林实现分类
'''

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import datetime
import numpy as np
import ML_GetData as getData
from ML04_DecisionTree_2 import DecisonTree 
import ML05_RandomForest_2

def RandomForest(X_train,y_train,X_test,y_test):
    starttime = datetime.datetime.now()
    #训练模型
    #n_estimator: 树的数量
    #bootstrap: 是否随机放回
    #n_jobs: 可并行运行的数量
    rfc = RandomForestClassifier(n_estimators=20,criterion = 'gini',bootstrap=True,n_jobs=2,random_state=8)
    rfc.fit(X_train,y_train)
    predict = rfc.predict(X_test)
    
    print('--------随机森林----------')
    print('预测结果准确率: %0.2f'%np.mean(predict==y_test.reshape(-1)))
    scores = cross_val_score(rfc, X_train, y_train,scoring='accuracy',cv=10)
    print('训练模型的准确率：%0.2f(+/-%0.2f)'%(scores.mean(),scores.std()*2))
    endtime = datetime.datetime.now()
    time_spend = endtime-starttime
    print('花费时间：%0.2f'%time_spend.total_seconds())

if __name__ == '__main__':
    X_train,X_test,y_train,y_test = getData.GetData('forest')    
    DecisonTree(X_train,y_train,X_test,y_test)
#     RandomForest(X_train, y_train, X_test, y_test)
    ML05_RandomForest_2.RF_PredictForestAreas_1(X_train, y_train, X_test, y_test)




