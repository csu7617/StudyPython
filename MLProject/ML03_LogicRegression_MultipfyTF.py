'''
Created on 2018年5月24日

@author: hw
'''
import pandas as pd
import tensorflow as tf
import numpy as np
from ML_GetData import GetData


def multiply_classfication_nntrain(input_data,input_answer,learning_rate,training_number,hide_layer_size):
    '''
    input_data: 输入
    input_answer: 输出
    learning_rate: 学习率
    training_numer: 训练次数
    hide_layer_size: 隐藏层大小
    '''
    #标签数目
    label_num = input_answer.shape[1]
    #特征数量
    feature_num = input_data.shape[1]
    
    X = tf.placeholder('float',[None,feature_num])
    y = tf.placeholder('float',[None,label_num])
    y_ = tf.placeholder('float', [None,label_num])
    W = []
    b = []
    if len(hide_layer_size) > 0:
        for i in range(len(hide_layer_size)):
            if i == 0:
                W.append(tf.Variable(tf.random_uniform([feature_num,hide_layer_size[i]],-1/np.sqrt(hide_layer_size[i]),1/np.sqrt(hide_layer_size[i]))))
                b.append(tf.Variable(tf.random_uniform([hide_layer_size[i]],-1/np.sqrt(hide_layer_size[i]),1/np.sqrt(hide_layer_size[i]))))
            else:
                W.append(tf.Variable(tf.random_uniform([hide_layer_size[i-1],hide_layer_size[i]],-1/np.sqrt(hide_layer_size[i]),1/np.sqrt(hide_layer_size[i]))))
                b.append(tf.Variable(tf.random_uniform([hide_layer_size[i]],-1/np.sqrt(hide_layer_size[i]),1/np.sqrt(hide_layer_size[i]))))
        W.append(tf.Variable(tf.random_uniform([hide_layer_size[len(hide_layer_size)-1],label_num],-1/np.sqrt(label_num),1/np.sqrt(label_num))))
        b.append(tf.Variable(tf.random_uniform([label_num],-1/np.sqrt(label_num),1/np.sqrt(label_num))))
    else:
        W.append(tf.Variable(tf.random_uniform([feature_num,label_num],-1/np.sqrt(label_num),1/np.sqrt(label_num))))
        b.append(tf.Variable(tf.random_uniform([label_num],-1/np.sqrt(label_num),1/np.sqrt(label_num))))
    #添加连接层
    layer_out = []
    #训练
    if len(hide_layer_size) > 0:
        for i in range(len(W)):
            if i==0:
                layer_out.append(tf.nn.relu(tf.add(tf.matmul(X,W[i]), b[i])))
            elif(i<len(W)-1):
                layer_out.append(tf.nn.relu(tf.add(tf.matmul(layer_out[i-1],W[i]),b[i])))
            else:
                y = tf.add(tf.matmul(layer_out[i-1],W[i]),b[i])
                y = tf.nn.softmax(y)
    else:
        y = tf.add(tf.matmul(X,W[0]),b[0])
        y = tf.nn.softmax(y)
    #将预测值转换为bool值
    y_bool = tf.equal(tf.argmax(y,axis=1),tf.argmax(y_,axis=1))
    accuracy = tf.reduce_mean(tf.cast(y_bool,'float'))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(training_number):
            sess.run(train,feed_dict={X:input_data,y_:input_answer})
            if i % 100 == 0:
                print(i,sess.run([loss,accuracy],feed_dict={X:input_data,y_:input_answer}))
        #保存模型
        saver = tf.train.Saver()
        save_path = saver.save(sess,'data/ML03_LR_tf_model.ckpt')
    tf.reset_default_graph()

#测试集
def multiply_classfication_nntest(input_data,input_answer,hide_layer_size):
    
    #标签数目
    label_num = input_answer.shape[1]
    #特征数目
    feature_num = input_data.shape[1]
 
    
    X = tf.placeholder('float',[None,feature_num])
    y = tf.placeholder('float',[None,label_num])
    y_ = tf.placeholder('float',[None,label_num])
    
    W = []
    b = []   
    
    if len(hide_layer_size) > 0:
        for i in range(len(hide_layer_size)):
            if i == 0:
                W.append(tf.Variable(tf.random_uniform([feature_num,hide_layer_size[i]],-1/np.sqrt(hide_layer_size[i]),1/np.sqrt(hide_layer_size[i]))))
                b.append(tf.Variable(tf.random_uniform([hide_layer_size[i]],-1/np.sqrt(hide_layer_size[i]),1/np.sqrt(hide_layer_size[i]))))
            else:
                W.append(tf.Variable(tf.random_uniform([hide_layer_size[i-1],hide_layer_size[i]],-1/np.sqrt(hide_layer_size[i]),1/np.sqrt(hide_layer_size[i]))))
                b.append(tf.Variable(tf.random_uniform([hide_layer_size[i]],-1/np.sqrt(hide_layer_size[i]),1/np.sqrt(hide_layer_size[i]))))
        W.append(tf.Variable(tf.random_uniform([hide_layer_size[len(hide_layer_size)-1],label_num],-1/np.sqrt(label_num),1/np.sqrt(label_num))))
        b.append(tf.Variable(tf.random_uniform([label_num],-1/np.sqrt(label_num),1/np.sqrt(label_num))))
    else:
        W.append(tf.Variable(tf.random_uniform([feature_num,label_num],-1/np.sqrt(label_num),1/np.sqrt(label_num))))
        b.append(tf.Variable(tf.random_uniform([label_num],-1/np.sqrt(label_num),1/np.sqrt(label_num))))
    #连接层
    layer_out = []
    if len(hide_layer_size) > 0:
        for i in range(len(W)):
            if i == 0:
                layer_out.append(tf.nn.relu(tf.add(tf.matmul(X,W[i]),b[i])))
            elif i < len(W)-1:
                layer_out.append(tf.nn.relu(tf.add(tf.matmul(layer_out[i-1],W[i]),b[i])))
            else:
                y = tf.add(tf.matmul(layer_out[i-1],W[i]),b[i])
                y = tf.nn.softmax(y)
    else:
       y = tf.add(tf.matmul(X,W[0]),b[0])
       y = tf.nn.softmax(y)
    
    #损失函数
#     loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y)
    #准确
    #将预测值转换为
    y_bool = tf.equal(tf.argmax(y,axis=1),tf.argmax(y_,axis=1))
    accuracy = tf.reduce_mean(tf.cast(y_bool,'float'))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess,'data/ML03_LR_tf_model.ckpt')
        print(sess.run(accuracy,feed_dict={X:input_data,y_:input_answer}))
    tf.reset_default_graph()
     

if __name__ == '__main__':
    
    #森林植被
    X_train,X_test,y_train,y_test =GetData('forest', 1)
    hide_layer_size = [8]
    multiply_classfication_nntrain(X_train, y_train, 0.01, 10000, hide_layer_size)
    multiply_classfication_nntest(X_test, y_test,hide_layer_size)