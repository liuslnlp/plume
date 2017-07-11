# plume
plume，一个轻量级机器学习库。包含常见机器学习算法的Python实现。


## 解决依赖
* Python >= 3.6
* NumPy
* SciPy


## 快速入门
对于训练集`X`，标签集合`y`，算法`clf`，测试集合`X_test`，使用`clf.fit(X, y)`训练数据，使用`clf.predcit(X_test)`对测试集进行预测。



Input:

```python
from plume.utils import plot_decision_boundary
from plume.svm import SVC

X, y = sklearn.datasets.make_moons(200, noise=0.20)
y = 2 * y - 1
clf = SVC(C=3, kernel='rbf')
clf.fit(X, y)
plot_decision_boundary(clf.predict, X, y, 'Support Vector Machine')
```
Output:

![img](docs/figure_svm.png)