# Machine Learning by Python
常见机器学习算法的Python实现。   
测试在Python3.6的环境下通过。

详细文档请[点击这里](https://wisedoge.github.io/ML-by-Python/mlearn.html)
## 解决依赖
* NumPy

## 使用方法
  所有的算法都重载了`fit(train_x, train_y), predict(test_x)`接口，使用方法类似sklearn。  
  例如：
```python
from mlearn.knn import KNeighborClassifier
import numpy as np
clf = KNeighborClassifier(n_neighbors=3)
train_x = np.array([[1, 1], [0.1, 0.1], [0.5, 0.7], [10, 10], [10, 11]])
train_y = np.array(['A', 'A', 'A', 'B', 'B'])
test_x = np.array([[11, 12], [12, 13], [11, 13], [0.05, 0.1]])
clf.fit(train_x, train_y)
print(clf.predict(test_x))
# 输出['B' 'B' 'B' 'A']

```

```python
from mlearn.naive_bayes import NaiveBayesClassifier
train_x = [["1", "S"], ["1", "M"], ["1", "M"],
           ["1", "S"], ["1", "S"], ["2", "S"],
           ["2", "M"], ["2", "M"], ["2", "L"],
           ["2", "L"], ["3", "L"], ["3", "M"],
           ["3", "M"], ["3", "L"], ["3", "L"]]
train_y = ["-1", "-1", "1", "1", "-1",
           "-1", "-1", "1", "1", "1",
           "1", "1", "1", "1", "-1"]
clf = NaiveBayesClassifier()
clf.fit(train_x, train_y)
print(clf.predict([["2", "S"]]))
# 输出['-1']
```

## 内容
* K近邻算法
* 感知机
* 朴素贝叶斯分类器
* 决策树(Decision Tree)
* 梯度提升树
* 随机森林(Random Forests)
* 支持向量机(Support Vector Machine)
* 线性回归(Linear Regression)
* 逻辑斯蒂回归(Logistic Regression)
* Bagging算法
* PageRank算法  
* 神经网络(BP算法)
* 隐马尔可夫模型
* K-Means聚类算法
* LVQ聚类算法
* 主成份分析法(PCA)


## 说明
本项目中的算法是作者在学习机器学习的过程中所做的练习，其中不免有些错误，如果有发现的错误，欢迎批评指正。  
