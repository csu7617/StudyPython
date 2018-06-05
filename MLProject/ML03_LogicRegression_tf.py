'''
Created on 2018年5月16日

@author: HW
'''

'''
用tensorflow实现二分类
'''


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

#定义全局变量
x = tf.placeholder('float',[None,2])
y = tf.placeholder('float',[None,1])            #预测值
y_= tf.placeholder('float',[None,1])            #结果
b = tf.Variable(tf.ones([1]))


W = tf.Variable(tf.ones([2,1]))
y = tf.add(tf.matmul(x,W),b)                             #用矩阵相乘

#定义loss函数
#     y = tf.nn.sigmoid(y)
#     loss = -tf.reduce_mean(tf.add(tf.multiply(y_,tf.log(y)),tf.multiply(1-y_,tf.log(1-y))))
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y))
optimizer = tf.train.AdamOptimizer(0.05)
train = optimizer.minimize(loss)
#将预测值转换为bool值
y_bool = tf.equal(tf.floor(tf.multiply(tf.nn.sigmoid(y),2)), y_)         #将预测值乘以2向下取整,即大于0.5取1，小于0.5取0
#计算准确率
accuracy = tf.reduce_mean(tf.cast(y_bool,'float'))

def GetData():
    csv_column_names = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
    train_data = pd.read_csv("data/iris_training.csv",names=csv_column_names,header = 0)
    test_data = pd.read_csv("data/iris_test.csv",names=csv_column_names,header=0)
    
    train_1 = train_data.iloc[:,[0,2]].values 
    train_2 = train_data.iloc[:,train_data.shape[1]-1:train_data.shape[1]].values
#     print(train_2[0][0])
    train_X = np.float32([ train_1[i][:] for i in range(train_1.shape[0]) if train_2[i][0] != 2])
    
    train_y = np.float32([ train_2[i][0] for i in range(train_2.shape[0]) if train_2[i][0] != 2 ]).reshape(-1,1)
    
    print(train_X,'\n',train_y)
    
    test_1 = test_data.iloc[:,[0,2]].values
    test_2 = test_data.iloc[:,test_data.shape[1]-1:test_data.shape[1]].values
    test_X = np.float32([ test_1[i][:] for i in range(test_1.shape[0]) if test_2[i][0] !=2 ])
    test_y = np.float32([ test_2[i][:] for i in range(test_2.shape[0]) if test_2[i][0] !=2 ])
    
    return train_X,train_y,test_X,test_y

#画图
def plot2(input_data,input_answer,loss_list,train_number,W_,b_):
    plot.figure(1)
    plot.subplot(1,2,1)
    #第一类的坐标
    x1 = [x1 for x1 in [input_data[i][0] for i in range(input_data.shape[0]) if input_answer[i] == 0]]
    y1 = [y1 for y1 in [input_data[i][1] for i in range(input_data.shape[0]) if input_answer[i] == 0]]
    plot.scatter(x1,y1,marker='x',color='b',label='1',s=15)
    #第二类的坐标
    x2 = [x2 for x2 in [input_data[i][0] for i in range(input_data.shape[0]) if input_answer[i] == 1]]
    y2 = [y2 for y2 in [input_data[i][1] for i in range(input_data.shape[0]) if input_answer[i] == 1]]
    plot.scatter(x2, y2, marker = 'o', color='r',label='1',s=15) 
    
    plotx = [input_data[i][0] for i in range(input_data.shape[0])]
    plotx.sort()
    plot.plot(plotx,(plotx*W_[0] + b_)/(-W_[1]))
    plot.xlabel("SepalLength")
    plot.ylabel("PetalLength")
    plot.legend(loc = 2)
    plot.title("Data Distribution Diagram")
    plot.subplot(1,2,2)
    plot.plot(list(range(train_number)),loss_list)
    plot.xlabel("Count")
    plot.ylabel("loss")
    plot.title("lossValue")
    plot.show()
    

#训练
def neural_network_training(input_data,input_answer,feature_size,learning_rate=0.01,training_number=10000):

    loss_list = []
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        for i in range(training_number):
            sess.run(train,feed_dict={x:input_data,y_:input_answer})
            loss_list.append(sess.run(loss,feed_dict={x:input_data,y_:input_answer}))
            if i%200 == 0:
                print(i,sess.run([loss,accuracy],feed_dict={x:input_data,y_:input_answer}))
        W_,b_ = sess.run([W,b],feed_dict={x:input_data,y_:input_answer})    
        saver_path = saver.save(sess, "data/ML03_LR_tf_model.ckpt")  # 将模型保存到save/model.ckpt文件
        plot2(input_data,input_answer,loss_list,training_number,W_,b_)
        
def nerual_network_test(input_data,input_answer):
    
#     saver = tf.train.import_meta_graph("data/ML03_LRtf_Model.ckpt.meta")
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'data/ML03_LR_tf_model.ckpt')
        print('test: ',sess.run(accuracy,feed_dict={x:input_data,y_:input_answer}))
    
if __name__ == '__main__':
    #获取数据
    train_X,train_y,test_X,test_y = GetData()
    neural_network_training(train_X, train_y, 2,0.001,1000)
    nerual_network_test(test_X, test_y)
    