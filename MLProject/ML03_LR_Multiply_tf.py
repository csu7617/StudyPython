'''
Created on 2018年5月17日

@author: HW
'''


import tensorflow as tf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.pyplot import subplot


def GetData():
    csv_column_names = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
    data_train = pd.read_csv("data//iris_training.csv",names=csv_column_names,header=0)
    data_test = pd.read_csv("data//iris_test.csv",names=csv_column_names,header = 0)
    train_1 = data_train.iloc[:,0:data_train.shape[1]-1].values
    train_2 = data_train.iloc[:,data_train.shape[1]-1:data_train.shape[1]].values
    train_X = np.matrix([ train_1[i][:] for i in range(train_1.shape[0]) ] )
    train_y = np.matrix([ train_2[i][:] for i in range(train_2.shape[0])])
    test_1 = data_test.iloc[:,0:data_test.shape[1]-1].values
    test_2 = data_test.iloc[:,data_test.shape[1]-1:data_test.shape[1]].values
    test_X = np.matrix([ test_1[i][:] for i in range(test_1.shape[0]) ])
    test_y = np.matrix([ test_2[i][:] for i in range(test_2.shape[0]) ])
    
    return train_X,train_y,test_X,test_y

def neural_network_train_multiply(input_data,input_answer,learning_rate=0.01,train_number=20000):
    feature_size = input_data.shape[1]
    print(feature_size)
    x = tf.placeholder('float',shape=[None,feature_size])
    y = tf.placeholder('float',shape=[None,3])
    y_ = tf.placeholder('float',shape=[None,3])
    
    W = tf.Variable(tf.random_normal([feature_size,3]))
    b = tf.Variable(tf.random_normal([3]))
    
    y = tf.nn.softmax((tf.matmul(x, W)+b))   #  不知道直接加是否可以
    #损失函数,用目标函数的分类和预测模型分类之间的交叉熵
#     cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y))
    cross_entropy = -tf.reduce_mean(y_*tf.log(y))
    
    #开始训练模型
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(train_number):
            sess.run(train_step,feed_dict={x:input_data,y_:input_answer})
            if i%200 == 0:
                print(sess.run([cross_entropy,accuracy],feed_dict={x:input_data,y_:input_answer}))
        W_,b_ = sess.run([W,b],feed_dict={x:input_data,y_:input_answer})
        
        
if __name__ == '__main__':
    train_X,train_y,test_X,test_y = GetData()
    
    train_y_modify = []
    for i in range(train_y.shape[0]):
        temp = []
        if train_y[i][0] == 0:
            temp.append(1)
            temp.append(0)
            temp.append(0)
            train_y_modify.append(temp)
        elif train_y[i][0] == 1:
            temp.append(0)
            temp.append(1)
            temp.append(0)
            train_y_modify.append(temp)
        else:
            temp.append(0)
            temp.append(0)
            temp.append(1)
            train_y_modify.append(temp)
    print(train_y_modify)
    
    
    neural_network_train_multiply(train_X,train_y_modify,0.009,10000)