'''
Created on 2018年6月4日

@author: HW
'''





import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances


'''
预测
'''
def predict(rating, similarity, type='user'):
    if type == 'user':
        mean_user_rating = rating.mean(axis = 1)
        rating_diff = (rating - mean_user_rating[:,np.newaxis])
        pred = mean_user_rating[:,np.newaxis] + similarity.dot(rating_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = rating.dot(similarity) / np.array([np.dot(similarity).sum(axis=1)])
    
    return pred



header = ['user_id','item_id','rating','timestamp']
df = pd.read_csv('data/ml-100k/u.data',sep='\t',names=header)

print(df)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

print('Number of users = ',str(n_users),' | Number of moives = ',str(n_items))

train_data,test_data = train_test_split(df, test_size = 0.3)

'''
基于内存协同过滤法
'''
train_data_matrix = np.zeros((n_users,n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1,line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1,line[2]-1] = line[3]
'''
使用pairwise distance函数来计算余弦相似性
'''

user_similarity = pairwise_distances(train_data_matrix,metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T,metric='cosine')

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')


    
    
    
    
    
    
    