'''
Created on 2018年5月19日

@author: HW
'''

'''
决策树算法在谷歌的莺尾花数据集上实现分类，并用jupyter notebook来查看所生成的树
'''

from sklearn import tree
from sklearn.datasets import load_iris
from matplotlib import pyplot as plot
from sklearn.model_selection import train_test_split
import pydotplus
from IPython.display import Image,display

iris = load_iris()
#初始化决策树模型
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data,iris.target)

#将模型存入dot文件iris.dot
with open("data/iris.dot",'w') as f:
    f = tree.export_graphviz(clf,out_file=f)

dot_data = tree.export_graphviz(clf,out_file=None,feature_names=iris.feature_names,class_names = iris.target_names,filled=True,rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('data/play1.pdf')