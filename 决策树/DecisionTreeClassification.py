'''
决策树分类的实现
用了ID3，C4.5算法
'''

import  numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split

class DecisionTreeClassification():
    def __init__(self, feature_names, eps=0.03, depth=10, min_leaf_size=1, method="gain"):
        '''
        eps:精度
        depth:深度
        min_leaf_size:最小叶子大小
        method:方法gain或者ratio
        '''
        self.feature_names = feature_names
        self.eps = eps
        self.depth = depth
        self.min_leaf_size=min_leaf_size
        self.method = method


    def gain_entropy(self, x):
        """
        计算信息增益
        x: 数据集的某个特征
        :return ent: H(D)=-\sum_{k=1}^K\frac{|C_k|}{|D|}\log_2\frac{|C_k|}{|D|}
        """
        entropy = 0
        for x_value in set(x):
            p = x[x == x_value].shape[0] / x.shape[0]
            entropy -= p * np.log2(p)
        return entropy

    def gain_condition_entropy(self, x, y):
        '''计算条件熵'''
        entropy = 0
        for x_value in set(x):
            sub_y = y[x == x_value]
            tmp_ent = self.gain_entropy(sub_y)
            p = sub_y.shape[0] / y.shape[0]
            entropy += p * tmp_ent
        return entropy


    def Gain(self, x, y):
        if self.method == "gain":
            return self.gain_entropy(x) - self.gain_condition_entropy(x, y)
        else:
            return 1 - self.gain_condition_entropy(x, y) / self.gain_entropy(x)

    def fit(self, X, y):
       self.tree = self._built_tree(X, y)

    def _built_tree(self, X, y, depth=1):
        # 只有一种标签了直接返回
        if len(set(y)) == 1:
            return y[0]

        # max_label：选择数量多的做代表标签
        label_1, label_2 = set(y)
        max_label = label_1 if np.sum(y == label_1) > np.sum(y == label_2) else label_2

        if len(X[0]) == 0:
            # 只剩下一种特征了
            return max_label

        if depth > self.depth:
            # 超过最大深度
            return max_label

        if len(y) < self.min_leaf_size:
            # 小于最小树叶大小
            return max_label


        best_feature_index = 0 # 最佳分类特征
        max_gain = 0 # 最大增益
        for feature_index in range(len(X[0])):
            gain = self.Gain(X[:, feature_index], y)
            if max_gain < gain:
                max_gain = gain
                best_feature_index = feature_index

        if max_gain < self.eps:
            # 最大增益比精度小
            return max_label

        # 用字典来保存树
        T = {}
        sub_T = {}
        for best_feature in set(X[:, best_feature_index]):
            '''
            best_feature：某个最佳特征下的特征类别
            '''
            sub_y = y[X[:, best_feature_index] == best_feature]
            sub_X = X[X[:, best_feature_index] == best_feature]
            sub_X = np.delete(sub_X, best_feature_index, 1) # 删除最佳特征列

            sub_T[best_feature+"__"+str(len(sub_X))] = self._built_tree(sub_X, sub_y, depth+1) # 关键代码

        T[self.feature_names[best_feature_index]+"__"+str(len(X))] = sub_T # 关键代码

        return T

    def predict(self, x, tree=None):
        if x.ndim == 2:
            res = []
            for x_ in x:

                res.append(self.predict(x_))
            return res

        if not tree:
            tree = self.tree

        tree_key = list(tree.keys())[0]

        x_feature = tree_key.split("__")[0]

        x_index = self.feature_names.index(x_feature) # 从列表中定位索引
        x_tree = tree[tree_key]
        for key in x_tree.keys():
            if key.split("__")[0] == x[x_index]:
                tree_key = key
                x_tree = x_tree[tree_key]

        if type(x_tree) == dict:
            return self.predict(x, x_tree)
        else:
            return x_tree


if __name__ == "__main__":
    df = pd.read_csv("/mnt/hgfs/Ubuntu/统计学习方法笔记/CH05/Input/mdata_5-1.txt", index_col=0)

    cols = df.columns.tolist()  # 特征名称列表
    X = df[cols[:-1]].values
    y = df[cols[-1]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = DecisionTreeClassification(feature_names=cols, eps=0.03, depth=5, min_leaf_size=1, method='ratio')
    clf.fit(X_train, y_train)

    print(clf.tree)
    # print(X_test)
    print("Result:")
    print("Real:")
    print(y_test)
    print("predict:")
    print(clf.predict(X_test))





