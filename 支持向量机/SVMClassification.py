'''
SVM分类, SMO算法实现
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class SVMClassification():
    def __init__(self, max_itr=100):
