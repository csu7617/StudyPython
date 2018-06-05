'''
Created on 2018年5月24日

@author: HW
'''

import pandas as pd
import tensorflow as tf
import numpy as np
from ML_GetData import GetData


'''
hide_layer_size: [100,50]
0 [1.864359, 0.5383297]
100 [1.4248773, 0.74138695]
200 [1.4094496, 0.75682306]
300 [1.4053751, 0.7600834]
400 [1.3982043, 0.768008]
500 [1.3964117, 0.76948816]
600 [1.3923646, 0.77376395]
700 [1.3901299, 0.7756621]
800 [1.390307, 0.7756671]
900 [1.3863354, 0.7795421]
1000 [1.3849368, 0.7808772]
1100 [1.3850162, 0.7804543]
1200 [1.3826115, 0.7831442]
1300 [1.3862108, 0.7790036]
1400 [1.3828994, 0.7828319]
1500 [1.3792217, 0.78658646]
1600 [1.3795651, 0.7860332]
1700 [1.3826576, 0.7826475]
1800 [1.3786614, 0.7869381]
1900 [1.378797, 0.7868053]
2000 [1.3799269, 0.7852833]
2100 [1.3785517, 0.7869749]
2200 [1.3774984, 0.7878355]
2300 [1.3772724, 0.7880002]
2400 [1.38114, 0.78384495]
2500 [1.3756202, 0.78977054]
2600 [1.3737007, 0.79191214]
2700 [1.3762386, 0.7890993]
2800 [1.3787898, 0.7863209]
2900 [1.3714749, 0.7939283]
3000 [1.373711, 0.7915974]
3100 [1.3717465, 0.79377586]
3200 [1.3762307, 0.7888387]
3300 [1.3707253, 0.7947053]
3400 [1.3752431, 0.7899107]
3500 [1.3721625, 0.79313415]
3600 [1.370425, 0.7950274]
3700 [1.3740724, 0.79102206]
3800 [1.3691895, 0.7961633]
3900 [1.3714586, 0.7937857]
4000 [1.3675599, 0.7978771]
4100 [1.3696009, 0.79563224]
4200 [1.3711647, 0.79397506]
4300 [1.3707726, 0.79427254]
4400 [1.3694686, 0.79579943]
4500 [1.3673402, 0.7979386]
4600 [1.3678288, 0.79746157]
4700 [1.3669219, 0.7984131]
4800 [1.3654891, 0.79987115]
4900 [1.3677194, 0.79760665]
'''


'''
hide_layer_size: [500,100]
0 [1.6957836, 0.4208474]
100 [1.4735743, 0.6906208]
200 [1.4555663, 0.70951885]
300 [1.4450603, 0.7203842]
'''

'''
[500,200,50,30]
0 [1.8853338, 0.37035662]
100 [1.4267939, 0.73778975]
200 [1.4116018, 0.7531054]
300 [1.403757, 0.7611308]
400 [1.4038314, 0.76079154]
500 [1.3830311, 0.7818238]
600 [1.3764076, 0.7884723]
700 [1.3728472, 0.79203016]
800 [1.3661051, 0.79875]
900 [1.3793548, 0.78518003]
1000 [1.3609062, 0.8038888]
9800 [1.2852695, 0.8793754]
9900 [1.2865828, 0.87811893]
0.86417407

'''

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

