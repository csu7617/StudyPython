'''
Created on 2018年5月17日

@author: Administrator
'''
import pandas as pd
import tensorflow as tf
import numpy as np
import gzip
from sklearn.model_selection import train_test_split

#对数据最好进行归一化处理，即进行特征缩放

#权重W和偏差b的初始化最好使用tf.random_uniform(shape,min,max),其中min，max 分别为正负1除以该层神经元个数的平方根


def get_data_from_gzfile(gz_file_path):
    input_data = []
    input_answer = []
    origin_answer = []
    with gzip.open(gz_file_path,'rt') as pf:
        for line in pf:
            line = line.strip()
            input_data.append([int(line.split(",")[i]) for i in range(len(line.split(","))-1)])
            origin_answer.append([int(line.split(",")[len(line.split(","))-1])])
    origin_answer = np.array(origin_answer).astype(np.float32)
    input_data = np.array(input_data).astype(np.float32)
    input_data = (input_data - np.mean(input_data, axis =0))/np.std(input_data,axis = 0)
    exist_class = list(set(origin_answer.reshape(-1).tolist()))
    for i in range(origin_answer.shape[0]):
        temp = []
        for current_class in exist_class:
            if origin_answer[i][0]==current_class:
                temp.append(1)
            else:
                temp.append(0)
        input_answer.append(temp)
    input_answer = np.array(input_answer)
    return input_data,input_answer



#从文件中获取数据，前面的所有列是特征，最后一列是分类结果
def get_data_from_csvfile(csv_file_path):
    data = pd.read_csv(csv_file_path)
    input_data = np.array(data.iloc[:,0:data.shape[1]-1].values).astype(np.float32)
    origin_answer = np.array(data.iloc[:,data.shape[1]-1:data.shape[1]].values).astype(np.float32)
    exist_class = list(set(origin_answer.reshape(-1).tolist()))
    input_answer = []
    for i in range(origin_answer.shape[0]):
        temp = []
        for current_class in exist_class:
            if origin_answer[i][0]==current_class:
                temp.append(1)
            else:
                temp.append(0)
        input_answer.append(temp)
    input_answer = np.array(input_answer)
#     print(input_answer.reshape(-1))
    input_data = (input_data - np.mean(input_data, axis =0))/np.std(input_data,axis = 0)
    return input_data,input_answer

def multiply_classfication_neural_network_train(input_data,input_answer,learning_rate,training_number,model_save_path,hide_layer_output_size_list):
#     pass
    class_num = input_answer.shape[1]
    feature_size = input_data.shape[1]
    x = tf.placeholder('float', [None,feature_size])
    y = tf.placeholder('float',[None,class_num])
    y_ = tf.placeholder('float',[None,class_num])#方便测试计算准确率
#     accuracy = tf.reduce_mean(tf.cast(tf.equal(y_,y),'float'))
    W = []
    b = []
    if(len(hide_layer_output_size_list)>0):
        for i in range(len(hide_layer_output_size_list)):
            if(i==0):
                W.append(tf.Variable(tf.random_uniform([feature_size,hide_layer_output_size_list[i]], -1/np.sqrt(hide_layer_output_size_list[i]), 1/np.sqrt(hide_layer_output_size_list[i]))))
                b.append(tf.Variable(tf.random_uniform([hide_layer_output_size_list[i]], -1/np.sqrt(hide_layer_output_size_list[i]), 1/np.sqrt(hide_layer_output_size_list[i]))))
            else:
                W.append(tf.Variable(tf.random_uniform([hide_layer_output_size_list[i-1],hide_layer_output_size_list[i]], -1/np.sqrt(hide_layer_output_size_list[i]), 1/np.sqrt(hide_layer_output_size_list[i]))))
                b.append(tf.Variable(tf.random_uniform([hide_layer_output_size_list[i]], -1/np.sqrt(hide_layer_output_size_list[i]), 1/np.sqrt(hide_layer_output_size_list[i]))))
        
        W.append(tf.Variable(tf.random_uniform([hide_layer_output_size_list[len(hide_layer_output_size_list)-1],class_num], -1/np.sqrt(class_num), 1/np.sqrt(class_num))))
        b.append(tf.Variable(tf.random_uniform([class_num], -1/np.sqrt(class_num), 1/np.sqrt(class_num))))
    else:
        W.append(tf.Variable(tf.random_uniform([feature_size,class_num], -1/np.sqrt(class_num), 1/np.sqrt(class_num))))
        b.append(tf.Variable(tf.random_uniform([class_num], -1/np.sqrt(class_num), 1/np.sqrt(class_num))))
    layer_out = []
    if len(hide_layer_output_size_list)>0:
        for i in range(len(W)):
            if(i==0):
                layer_out.append(tf.nn.relu(tf.add(tf.matmul(x,W[i]),b[i])))
            elif(i<len(W)-1):
                layer_out.append(tf.nn.relu(tf.add(tf.matmul(layer_out[i-1],W[i]),b[i])))
            else:
                y = tf.add(tf.matmul(layer_out[i-1],W[i]),b[i])
                y = tf.nn.softmax(y)
    else:
        y = tf.add(tf.matmul(x,W[0]),b[0])
        y = tf.nn.softmax(y)
#     W = tf.Variable(tf.random_normal([feature_size,class_num]))
#     b = tf.Variable(tf.random_normal([class_num]))
#     W = tf.Variable(tf.ones([feature_size,class_num]))
#     b = tf.Variable(tf.ones([class_num]))
#     lamda = 0.01
#     y = tf.add(tf.matmul(x,W),b)
#     y = tf.nn.softmax(y)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,axis=1),tf.argmax(y_,axis=1)),'float'))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
#     loss = loss + tf.multiply(lamda,tf.reduce_sum(W))
#     loss = -tf.reduce_mean(y_*tf.log(y))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for j in range(training_number):
            sess.run(train,feed_dict={x:input_data,y_:input_answer})
            if(j%50==0):
                print(j*50,sess.run([loss,accuracy],feed_dict={x:input_data,y_:input_answer}))
        if(model_save_path!=''):
            #保存模型
            saver = tf.train.Saver()
    #         model_path = "D:\sample\model.ckpt"
            save_path = saver.save(sess, model_save_path)
    tf.reset_default_graph()
    
    
def multiply_classfication_neural_network_test(test_data,test_answer,model_restore_path,hide_layer_output_size_list):
    class_num = test_answer.shape[1]
    feature_size = test_data.shape[1]
    x = tf.placeholder('float', [None,feature_size])
    y = tf.placeholder('float',[None,class_num])
    y_ = tf.placeholder('float',[None,class_num])#方便测试计算准确率
#     accuracy = tf.reduce_mean(tf.cast(tf.equal(y_,y),'float'))
    W = []
    b = []
    if(len(hide_layer_output_size_list)>0):
        for i in range(len(hide_layer_output_size_list)):
            if(i==0):
                W.append(tf.Variable(tf.random_uniform([feature_size,hide_layer_output_size_list[i]], -1/np.sqrt(hide_layer_output_size_list[i]), 1/np.sqrt(hide_layer_output_size_list[i]))))
                b.append(tf.Variable(tf.random_uniform([hide_layer_output_size_list[i]], -1/np.sqrt(hide_layer_output_size_list[i]), 1/np.sqrt(hide_layer_output_size_list[i]))))
            else:
                W.append(tf.Variable(tf.random_uniform([hide_layer_output_size_list[i-1],hide_layer_output_size_list[i]], -1/np.sqrt(hide_layer_output_size_list[i]), 1/np.sqrt(hide_layer_output_size_list[i]))))
                b.append(tf.Variable(tf.random_uniform([hide_layer_output_size_list[i]], -1/np.sqrt(hide_layer_output_size_list[i]), 1/np.sqrt(hide_layer_output_size_list[i]))))
        
        W.append(tf.Variable(tf.random_uniform([hide_layer_output_size_list[len(hide_layer_output_size_list)-1],class_num], -1/np.sqrt(class_num), 1/np.sqrt(class_num))))
        b.append(tf.Variable(tf.random_uniform([class_num], -1/np.sqrt(class_num), 1/np.sqrt(class_num))))
    else:
        W.append(tf.Variable(tf.random_uniform([feature_size,class_num], -1/np.sqrt(class_num), 1/np.sqrt(class_num))))
        b.append(tf.Variable(tf.random_uniform([class_num], -1/np.sqrt(class_num), 1/np.sqrt(class_num))))
    layer_out = []
    if len(hide_layer_output_size_list)>0:
        for i in range(len(W)):
            if(i==0):
                layer_out.append(tf.nn.relu(tf.add(tf.matmul(x,W[i]),b[i])))
            elif(i<len(W)-1):
                layer_out.append(tf.nn.relu(tf.add(tf.matmul(layer_out[i-1],W[i]),b[i])))
            else:
                y = tf.add(tf.matmul(layer_out[i-1],W[i]),b[i])
                y = tf.nn.softmax(y)
    else:
        y = tf.add(tf.matmul(x,W[0]),b[0])
        y = tf.nn.softmax(y)
#     loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = input_answer, logits = y)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,axis=1),tf.argmax(y_,axis=1)),'float'))
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, model_restore_path)
        p = sess.run(accuracy, feed_dict={x: test_data,y_:test_answer})
#         predict = sess.run(tf.argmax(y,axis=1),feed_dict={x:test_data})
    tf.reset_default_graph()
    return p

if __name__=='__main__':
    train_file_path = 'data/iris_training.csv'
    test_file_path = 'data/iris_test.csv'
    
    gz_file_path = 'data/covtype/covtype.data.gz'
    
#森林植被
    train_x,train_y = get_data_from_gzfile(gz_file_path)
    train_data,test_data,train_answer,test_answer = train_test_split(train_x,train_y,test_size=0.3,random_state=2)

#  iris
#     train_data,train_answer = get_data_from_csvfile(train_file_path)
#     test_data,test_answer = get_data_from_csvfile(test_file_path)
#     print(train_answer)

#手写识别测试
#     mnist_data = input_data.read_data_sets("data/",one_hot=True)
#     train_data = mnist_data.train.images
#     train_answer = mnist_data.train.labels
#     test_data = mnist_data.test.images
#     test_answer = mnist_data.test.labels
    
    learning_rate = 0.01
    training_number = 200
    model_save_path = 'data/model.ckpt'
    
    hide_layer_output_size_list = [500,200,50]
    
    
    multiply_classfication_neural_network_train(train_data,train_answer,learning_rate,training_number,model_save_path,hide_layer_output_size_list)
    accuracy = multiply_classfication_neural_network_test(test_data,test_answer,model_save_path,hide_layer_output_size_list)
    print("test accuracy=",accuracy)
    
    
    