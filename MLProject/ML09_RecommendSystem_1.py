'''
Created on 2018年6月4日

@author: HW
'''

'''
利用两个用户电影评分的向量进行相似度计算，寻找邻居进行预测
'''


from matplotlib import pyplot
import scipy as sp
import numpy as np
from ML_GetData import readFile
import sys
from audioop import reverse
# from texttable import Texttable

#计算余弦距离
def getCosDist(user1, user2):
    sumx = 0.0
    sumy = 0.0
    sumxy = 0.0
    for k1 in user1:
        for k2 in user2:
            if k1[0] == k2[0]:
                sumx += k1[1]*k1[1]
                sumy += k2[1]*k2[1]
                sumxy += k1[1]*k2[1]
    if sumxy == 0.0:
        return 0
    demo = np.sqrt(sumx * sumy)
    return sumxy / demo

def getMoiveList(filename):
    contents = []
    with open(filename,mode='r+',encoding='utf-8') as f:
        for line in f:
            contents.append(line)
    
    moive_info = {}
    for moive in contents:
        single_info = moive.split("|")
        moive_info[int(single_info[0])] = single_info[1:len(single_info)]
    return moive_info

#生成用户评分数据结构
def getUserScoreDataStructure(contents):
    userDict = {}
    itemUser = {}
    #userDict[2] = [(1,5),(4,2)]  表示用户2对电影1的评分为5，对电影4的评分为2
    for k in contents:
        #评分最高取5，除以5进行归一化
        user_rank = (k[1],float(k[2])/5)
        if k[0] in userDict:
            userDict[k[0]].append(user_rank)
        else:
            userDict[k[0]] = [user_rank]
        if k[1] in itemUser:
            itemUser[k[1]].append(k[0])
        else:
            itemUser[k[1]] = [k[0]]
#     item = userDict[1]
#     item_0 = itemUser[item[0]]
#     print(item,item_0)
    return userDict, itemUser

#计算与指定用户最相近的邻居
def getNearestNeighbor(userId,userDict,itemUser):
    neighbors = []
    for item in userDict[userId]:
        for neighbor in itemUser[item[0]]:
            if neighbor != userId and neighbor not in neighbors:
                neighbors.append(neighbor)
    neighbor_dist = []
    for neighbor in neighbors:
        dist = getCosDist(userDict[userId],userDict[neighbor])
        neighbor_dist.append([dist,neighbor])
    neighbor_dist.sort(reverse=True)

    return neighbor_dist
    
#使用UserFC进行推荐，输入文件名，用户ID，邻居数量
def recommandByUserFC(filename, userId, k = 5):
    #读取文件
    contents = readFile(filename, '\t', 3, encoding='utf-8',DataStyle='int')
    contents = np.array(contents)
    #格式化数据字典
    userDict, itemUser = getUserScoreDataStructure(contents)
    #找邻居
    neighbors = getNearestNeighbor(userId,userDict,itemUser)[0:5]
#     print(neighbors)
    #建立推荐字典
    recommand_dict = {}
    for neighbor in neighbors:
        neighbor_user_id = neighbor[1]
        moives = userDict[neighbor_user_id]
        for moive in moives:
            if moive[0] not in recommand_dict:
                recommand_dict[moive[0]] = neighbor[0]
            else:
                recommand_dict[moive[0]] += neighbor[0] 
    #建立推荐列表
    recommand_list = []
    for key in recommand_dict:
        recommand_list.append([recommand_dict[key],key])
    recommand_list.sort(reverse=True)
    user_moives = [k[0] for k in userDict[userId]]
    return [k[1] for k in recommand_list[0:10]],user_moives,itemUser,neighbors
    
    
    
if __name__ == '__main__':    
    moives = getMoiveList('data/ml-100k/u.item')
    recommand_list,user_moive,itemUser,neighbors = recommandByUserFC('data/ml-100k/u.data', 100000, 5)
    neighbors_id = [id[1] for id in neighbors]
    print(recommand_list)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     table = Texttable()
#     table.set_deco(Texttable.HEADER)
#     table.set_cols_dtype(['t','t','t'])
#     table.set_cols_align(['1','1','1'])
#     rows = []
#     rows.append([u'moive name',u'release',u'from userid'])
#     for moive_id in recommand_list[:20]:
#         from_user = []
#         for user_id in itemUser[moive_id]:
#             if user_id in neighbors_id:
#                 from_user.append(user_id)
#         row = []
#         row.append(moives[moive_id][0])
#         row.append(moives[moive_id][1])
#         row.append("")
#         rows.append(row)
#     table.add_row(rows)
#     print(table.draw())
    

    
    
    
    
    
    
    