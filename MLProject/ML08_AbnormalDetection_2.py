'''
Created on 2018年6月1日

@author: HW
'''

import numpy as np
import random
import ML08_AbnormalDeetection as ad

def GetData(train_num,wrong_num):
    x1 = [i*1 for i in range(train_num)]
    x1 = np.array(x1)
#     x1 = x1.reshape(-1,1)
    x2 = 2*x1 + 1.5
    
    label1 = np.ones([train_num,1])
    x = np.c_[x1,x2,label1]
    #打乱顺序
    x = x[random.sample(range(train_num),train_num),:]
    x1_wrong = np.array(random.sample(x1.tolist(),wrong_num))
    rand_bias = np.array(random.sample([i/10 for i in range(-10,10)],wrong_num))
    x2_wrong = 2*x1_wrong + rand_bias
    label2 = np.zeros(([wrong_num,1]))
    x_wrong = np.c_[x1_wrong,x2_wrong,label2]
    #
    x_wrong = x_wrong[random.sample(range(wrong_num),wrong_num),:]
    return x,x_wrong

def CV(X_validation,X_wrong,u,sigma2):
    p_normal = []
    p_wrong = []
    minp_normal = 1
    maxp_wrong = 0
    for i in range(X_validation.shape[0]):
        temp = ad.AbnormalDetection(u, sigma2, X_validation[i])
        p_normal.append(temp)
        if minp_normal > temp:
            minp_normal = temp
    for i in range(X_wrong.shape[0]):
        temp = ad.AbnormalDetection(u, sigma2, X_wrong[i])
        p_wrong.append(temp)
        if maxp_wrong < temp:
            maxp_wrong = temp
    print(minp_normal,'\n------------\n',maxp_wrong)
    return minp_normal,maxp_wrong

def Test(X_test,label_test,X_wrong,label_wrong,u,sigma2,bv):
    accuracy_normal = []
    accuracy_wrong = []
    for i in range(X_test.shape[0]):
        temp = ad.AbnormalDetection(u, sigma2, X_test[i])
        if temp > bv:
            accuracy_normal.append(1.0)
        else:
            accuracy_normal.append(0.0)
#     print(label_test,accuracy_normal)
    print(np.mean(label_test.reshape(-1)==accuracy_normal))
    for i in range(X_wrong.shape[0]):
        temp = ad.AbnormalDetection(u,sigma2,X_wrong[i])
        if temp < bv:
            accuracy_wrong.append(0)
        else:
            accuracy_wrong.append(1)
    print(np.mean(label_wrong.reshape(-1)==accuracy_wrong))

if __name__ == '__main__':
    x,x_wrong = GetData(100,10)
    print(x,x_wrong)
    #拆分验证集和测试集
    X_validation = x[0:60,:]
    X_test = x[60:100,:]    
    X_wrong_1 = x_wrong[0:5,:]
    X_wrong_2 = x_wrong[5:10,:]
    
    u = ad.getU(X_validation[:,0:2])
    sigma2 = ad.getSigma2(X_validation[:,0:2], u)
    minp_normal,maxp_wrong = CV(X_validation[:,0:2], X_wrong_1[:,0:2], u, sigma2)
    bv = (minp_normal+maxp_wrong)/2
    Test(X_test[:,0:2],X_test[:,2:3],X_wrong_2[:,0:2],X_wrong_2[:,2:3],u,sigma2,bv)