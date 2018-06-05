'''
Created on 2018年5月22日

@author: hw
'''

'''
使用随机森林预测森林植被
'''
from sklearn.grid_search import GridSearchCV
import datetime
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np


'''
------------随机森林预测准确率----------
测试集准确率: 0.84
训练集准确率: 0.84(+/- 0.00)
花费时间: 366.87

'''
#默认方式解决问题
def RF_PredictForestAreas_2(X_train,y_train,X_test,y_test):
    starttime = datetime.datetime.now()
    #训练模型
    rfc = RandomForestClassifier(n_estimators=50,min_samples_leaf=50,criterion='gini',bootstrap=True,random_state=8)
    rfc.fit(X_train,y_train)
    
    predict = rfc.predict(X_test)
    print('------------随机森林预测准确率----------')
    print('测试集准确率: %0.2f'%np.mean(predict==y_test.reshape(-1)))
    scores = cross_val_score(rfc,X_train,y_train,scoring='accuracy',cv=10)
    print('训练集准确率: %0.2f(+/- %0.2f)'%(scores.mean(),scores.std()*2))
    endtime = datetime.datetime.now()
    spendtime = endtime-starttime
    print('花费时间: %0.2f'%spendtime.total_seconds())




#自行选择最优组合
'''
------随机森林预测准确率--------
测试集准确率: 0.84
训练集准确率：0.84(+/- 0.00)
花费时间：1717.53
'''
def RF_PredictForestAreas_1_1(X_train,y_train,X_test,y_test):
    starttime = datetime.datetime.now()
    #训练模型
    #n_estimators: 表示树的数量
    rfc = RandomForestClassifier(criterion='gini',bootstrap=True,random_state=8)
    #需要选择的参数名称和候选值
    tuned_parameter = [{'min_samples_leaf': [50], 'n_estimators':[50]}]
    #cv交叉验证
    clf = GridSearchCV(estimator=rfc,param_grid=tuned_parameter,cv = 5,n_jobs=1)
    #拟合训练集
    clf.fit(X_train,y_train)
    
    predict = clf.predict(X_test)
    
    print('------随机森林预测准确率--------')
    print('测试集准确率: %0.2f'%np.mean(predict == y_test.reshape(-1)))
    scores = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv = 10)
    print('训练集准确率：%0.2f(+/- %0.2f)'%(scores.mean(),scores.std()*2))
    endtime = datetime.datetime.now()
    spendtime = endtime - starttime
    print('花费时间：%0.2f'%spendtime.total_seconds())
    
'''
------随机森林预测准确率--------
测试集准确率: 0.93
训练集准确率：0.93(+/- 0.00)
花费时间：3965.48
'''   
#自行选择最优组合
def RF_PredictForestAreas_1_2(X_train,y_train,X_test,y_test):
    starttime = datetime.datetime.now()
    #训练模型
    #n_estimators: 表示树的数量
    rfc = RandomForestClassifier(criterion='gini',bootstrap=True,random_state=8)
    #需要选择的参数名称和候选值
    tuned_parameter = [{'min_samples_leaf': [50,5], 'n_estimators':[50]}]
    #cv交叉验证
    clf = GridSearchCV(estimator=rfc,param_grid=tuned_parameter,cv = 5,n_jobs=1)
    #拟合训练集
    clf.fit(X_train,y_train)
    
    predict = clf.predict(X_test)
    
    print('------随机森林预测准确率--------')
    print('测试集准确率: %0.2f'%np.mean(predict == y_test.reshape(-1)))
    scores = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv = 10)
    print('训练集准确率：%0.2f(+/- %0.2f)'%(scores.mean(),scores.std()*2))
    endtime = datetime.datetime.now()
    spendtime = endtime - starttime
    print('花费时间：%0.2f'%spendtime.total_seconds())
    

'''
------随机森林预测准确率--------
测试集准确率: 0.95
训练集准确率：0.94(+/- 0.00)
花费时间：7858.24
'''
#自行选择最优组合
def RF_PredictForestAreas_1_3(X_train,y_train,X_test,y_test):
    starttime = datetime.datetime.now()
    #训练模型
    #n_estimators: 表示树的数量
    rfc = RandomForestClassifier(criterion='gini',bootstrap=True,random_state=8)
    #需要选择的参数名称和候选值
    tuned_parameter = [{'min_samples_leaf': [50,5,2,3], 'n_estimators':[50]}]
    #cv交叉验证
    clf = GridSearchCV(estimator=rfc,param_grid=tuned_parameter,cv = 5,n_jobs=1)
    #拟合训练集
    clf.fit(X_train,y_train)
    
    predict = clf.predict(X_test)
    
    print('------随机森林预测准确率--------')
    print('测试集准确率: %0.2f'%np.mean(predict == y_test.reshape(-1)))
    scores = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv = 10)
    print('训练集准确率：%0.2f(+/- %0.2f)'%(scores.mean(),scores.std()*2))
    endtime = datetime.datetime.now()
    spendtime = endtime - starttime
    print('花费时间：%0.2f'%spendtime.total_seconds())

'''
------随机森林预测准确率--------
测试集准确率: 0.95
训练集准确率：0.95(+/- 0.00)
花费时间：12406.39
'''
#自行选择最优组合
def RF_PredictForestAreas_1_4(X_train,y_train,X_test,y_test):
    starttime = datetime.datetime.now()
    #训练模型
    #n_estimators: 表示树的数量
    rfc = RandomForestClassifier(criterion='gini',bootstrap=True,random_state=8)
    #需要选择的参数名称和候选值
    tuned_parameter = [{ 'n_estimators':[50,10,20,100]}]
    #cv交叉验证
    clf = GridSearchCV(estimator=rfc,param_grid=tuned_parameter,cv = 5,n_jobs=1)
    #拟合训练集
    clf.fit(X_train,y_train)
    
    predict = clf.predict(X_test)
    
    print('------随机森林预测准确率--------')
    print('测试集准确率: %0.2f'%np.mean(predict == y_test.reshape(-1)))
    scores = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv = 10)
    print('训练集准确率：%0.2f(+/- %0.2f)'%(scores.mean(),scores.std()*2))
    endtime = datetime.datetime.now()
    spendtime = endtime - starttime
    print('花费时间：%0.2f'%spendtime.total_seconds())
    
'''
------随机森林预测准确率--------
测试集准确率: 0.95
训练集准确率：0.94(+/- 0.00)
花费时间：3809.94
'''
#自行选择最优组合
def RF_PredictForestAreas_1_5(X_train,y_train,X_test,y_test):
    starttime = datetime.datetime.now()
    #训练模型
    #n_estimators: 表示树的数量
    rfc = RandomForestClassifier(criterion='gini',bootstrap=True,random_state=8)
    #需要选择的参数名称和候选值
    tuned_parameter = [{'min_samples_leaf': [2], 'n_estimators':[50,10,20]}]
    #cv交叉验证
    clf = GridSearchCV(estimator=rfc,param_grid=tuned_parameter,cv = 5,n_jobs=1)
    #拟合训练集
    clf.fit(X_train,y_train)
    
    predict = clf.predict(X_test)
    
    print('------随机森林预测准确率--------')
    print('测试集准确率: %0.2f'%np.mean(predict == y_test.reshape(-1)))
    scores = cross_val_score(clf,X_train,y_train,scoring='accuracy',cv = 10)
    print('训练集准确率：%0.2f(+/- %0.2f)'%(scores.mean(),scores.std()*2))
    endtime = datetime.datetime.now()
    spendtime = endtime - starttime
    print('花费时间：%0.2f'%spendtime.total_seconds())
