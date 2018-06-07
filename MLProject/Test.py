'''
Created on 2018年5月14日

@author: Administrator
'''

import tensorflow as tf;  
import numpy as np;  
from sklearn import datasets
# from numpy import *
import random
import pandas as pd

path = "data/iris_test.csv"
#names添加列名，header用指定的行为作为标题，若无原标题则设为None
data = pd.read_csv(path,header=None,names= ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species'])
print(data)