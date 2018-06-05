'''
Created on 2018年5月21日

@author: HW
'''
import pydotplus
from builtins import str


'''
用决策树算法实现一个小案例(是否打网球)
'''
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import pydotplus
def outlook_type(s):
    s = s.decode('utf-8')
    it = {'sunny':1, 'overcast':2, 'rainy':3}
    return it[s]

def temperature(s):
    s = s.decode('utf-8')
    it = {'hot':1,'mild':2,'cool':3}
    return it[s]

def humidity(s):
    s = s.decode('utf-8')
    it = {'high': 1,'normal':0}
    return it[s]

def windy(s):
    s = s.decode('utf-8')
    it = {'TRUE':1,'FALSE':0}
    return it[s]

def play_type(s):
    s = s.decode('utf-8')
    it = {'yes':1,'no':0}
    return it[s]

play_feature_E = 'outlook','temperature','humidity','windy'
play_class = 'yes','no'

#读入数据,并将原始数据中的数据转换为数字形式
data= np.loadtxt('data/play_tennis.txt',delimiter=' ',dtype=float, converters={0:outlook_type, 1:temperature, 2:humidity, 3:windy, 4:play_type})

x,y = np.split(data,(4,),axis = 1)

#拆分训练数据与测试数据，为了进行交叉验证
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

#使用信息熵作为划分标准，对决策树进行训练
clf = tree.DecisionTreeClassifier(criterion='entropy')
print('clf',clf)
clf.fit(x_train, y_train)

#把决策树结构写入文件
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=play_feature_E,class_names=play_class,filled=True,rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('data/play1.pdf')

print(dot_data)
#系数反应每个特征的影响力
print('每个特征的影响力',clf.feature_importances_)
 
#使用训练数据做预测,预测结果完全正确
answer = clf.predict(x_train)
y_train = y_train.reshape(-1)
print('使用训练数据做预测: ',np.mean(answer == y_train))
 
#对测试数据进行预测,准确度较低，说明过拟合
answer = clf.predict(x_test)
y_test = y_test.reshape(-1)
print(answer)
print(y_test)
print('对测试数据进行预测: ',np.mean(answer == y_test))
