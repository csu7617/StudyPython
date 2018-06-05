'''
Created on 2018年5月30日

@author: HW
'''

from sklearn import svm
from ML_GetData import GetData
import numpy as np

def SVM(X_train,X_test,y_train,y_test):
    clf = svm.SVC(C=0.8,kernel='linear',gamma=20,decision_function_shape='ovr')
    clf.fit(X_train,y_train)
    predict_train = clf.predict(X_train)
    predict_test = clf.predict(X_test)
    print('训练集准确率：',np.mean(predict_train==y_train.reshape(-1)))
    print('测试集准确率： ',np.mean(predict_test==y_test.reshape(-1)))


if __name__ == '__main__':
    X_train,X_test,y_train,y_test = GetData('iris', 0)
    SVM(X_train, X_test, y_train, y_test)