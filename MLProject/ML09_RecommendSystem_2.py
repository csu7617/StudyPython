'''
Created on Jun 7, 2018

@author: hw
'''
import numpy as np
import tensorflow as tf
from builtins import len
import datetime

#offset 表示从哪开始读
def  read_movielens_data(filepath,split_flag,encoding,offset = 0 ,read_line_num=None):
    data = []
    with open(filepath,mode='r+',encoding = encoding) as f:
        f.seek(offset)
        read_i = 0
        while True:
            line = f.readline()
            if(line==''):
                break
            else:
                line = line.strip()
                content = line.split(split_flag)
                data.append([int(content[i]) for i in range(len(content)-1)])
                if read_line_num!=None:
                    read_i +=1
                    if read_i>=read_line_num:
                        break
        position = f.tell()
    data = np.array(data)
    
    
    
#     moive = data[:,1]
#     moive = list(set(moive.tolist()))
#     print(moive)
        
    return data,position

def data_process(data):
    u_user = np.zeros([943])#对每个人所评价过的电影的平均值
    u_movie = np.zeros([1682])#对同一部电影的所有人评分的平均值
    use_data = np.zeros([943,1682])#943人分别对1682部电影的评分减去相应的均值u_user,u_movie，未评分则为0
    for i in range(943):
        rating_index = np.argwhere(data[:,0]==i+1).reshape(-1)
        rating_array = data[rating_index,2].astype(np.float32)
        if(rating_array.shape[0]>0):
            u_user[i] = np.mean(rating_array)
    for i in range(1682):
        rating_index = np.argwhere(data[:,1]==i+1).reshape(-1)
        rating_array = data[rating_index,2].astype(np.float32)
        if(rating_array.shape[0]>0):
            u_movie[i] = np.mean(rating_array)
#     print(u_movie)
    for i in range(943):
        user_index = np.argwhere(data[:,0] == i+1).reshape(-1)

        user_data = data[user_index,:]
        for j in range(1682):
            moive_index = np.argwhere(user_data[:,1]==j+1).reshape(-1)
            if moive_index.shape[0]>0:
                use_data[i,j] = float(user_data[moive_index[0],2])-u_movie[j]-u_user[i]
#     print(use_data)
    return u_user,u_movie,use_data

def training(training_number,use_data,filter_predict,feature_size,learning_rate = 0.01,lamda = 0.01,min_loss = 0.01):
    
   
    starttime = datetime.datetime.now()
    use_data = np.array(use_data)
    use_data = use_data.reshape([use_data.shape[0],-1])
    x = tf.Variable(tf.truncated_normal([use_data.shape[0],feature_size],stddev=0.02))
#     x = tf.Variable(tf.random_uniform([use_data.shape[0],feature_size],-1/np.sqrt(feature_size),1/np.sqrt(feature_size)))
    w = tf.Variable(tf.truncated_normal([use_data.shape[1],feature_size],stddev=0.02))
#     w = tf.Variable(tf.random_uniform([use_data.shape[1],feature_size],-1/np.sqrt(feature_size),1/np.sqrt(feature_size)))
    predict = tf.matmul(x,tf.transpose(w))
    predict = tf.multiply(filter_predict, predict)
    loss = tf.reduce_sum(tf.square(predict-use_data))+lamda*tf.reduce_sum(tf.square(x)) + lamda*tf.reduce_sum(tf.square(w))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for train_step in range(training_number):
            sess.run(train)
            if train_step %200 == 0:
                print(sess.run(loss))
#             if(sess.run(loss)<=min_loss):
#                 break

        saver = tf.train.Saver()
        saver.save(sess, 'data/model/model_ml09_rs_1.ckpt')
#             print(sess.run(loss))
    tf.reset_default_graph()   
    endtime = datetime.datetime.now()
    
    print('花费时间：%f'%(endtime-starttime).total_seconds())
    
    
def test(use_data,filter_predict,feature_size,u_user,u_movie,which_user,recommend_num):
    
    use_data = np.array(use_data)
    use_data = use_data.reshape([use_data.shape[0],-1])
    
    x = tf.Variable(tf.truncated_normal([use_data.shape[0],feature_size],stddev=0.02))
#     x = tf.Variable(tf.random_uniform([use_data.shape[0],feature_size],-1/np.sqrt(feature_size),1/np.sqrt(feature_size)))
    w = tf.Variable(tf.truncated_normal([use_data.shape[1],feature_size],stddev=0.02))
#     w = tf.Variable(tf.random_uniform([use_data.shape[1],feature_size],-1/np.sqrt(feature_size),1/np.sqrt(feature_size)))
    predict = tf.matmul(x,tf.transpose(w))
    predict = tf.multiply(filter_predict, predict)
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, 'data/model/model_ml09_rs_1.ckpt')
        x_ = sess.run(x)
        w_ = sess.run(w)
    x = x_
    w = w_
    tf.reset_default_graph()
    
    #找到用户对电影所有的评分
    x_user = x[which_user-1,:]
    
    #找到未预测电影的index
    filter_predict_user = filter_predict[which_user-1,:]
    unrating_index = np.argwhere(filter_predict_user==0).reshape(-1)
    
    #进行预测
    predict = np.dot(w[unrating_index,:],np.transpose(x_user)).reshape(-1)
    
    
    predict = predict + u_user[which_user-1]+ u_movie[unrating_index]
    predict = predict[0:137]
    recommend = [] 
    max = 5
    for i in range(recommend_num):
        #找到预测中的最大值
        max_index = np.argmax(predict)
#         if i == 0:
#             max = predict[max_index]
        recommend.append({unrating_index[max_index]+1:(predict[max_index]/max) *5})
        #在预测值以及index中删除最大值
        predict = np.delete(predict,max_index)
        unrating_index = np.delete(unrating_index,max_index)
        if(unrating_index.shape[0]<=0):
            break
        
    for i in range(recommend_num):
        min_index = np.argmin(predict)
        recommend.append({unrating_index[min_index]+1:predict[min_index]})
        predict = np.delete(predict,min_index)
        unrating_index = np.delete(unrating_index,min_index)
        if(unrating_index.shape[0]<=0):
            break
    return recommend
            

if __name__=='__main__':
    
    training_number = 10000
    data_filepath = 'data/ml-100k/u1.base'
    split_flag = '\t'
    encoding = 'utf-8'
    
    data,position = read_movielens_data(data_filepath,split_flag,encoding)#userid,movieid,rating(评分)
    u_user,u_movie,use_data = data_process(data)#u_user,u_movie在预测时才会用到
    filter_predict = np.array([[0.0 if use_data[i,j]==0 else 1.0 for j in range(use_data.shape[1])] for i in range(use_data.shape[0])])#当use_data中的元素为0时，则这个矩阵相应位置为0，其余位置为1
    filter_predict =filter_predict.astype(np.float32)
    print(filter_predict.shape)
    training(training_number,use_data,filter_predict,feature_size=5,learning_rate = 0.05,lamda = 0.1,min_loss = 0.1)
    
    recommend = test(use_data,filter_predict,5,u_user,u_movie,1,5)
    print(recommend)