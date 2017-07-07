# plume
plume，一个轻量级机器学习库，内含常见机器学习算法的Python实现。

使用的Python版本必须在*3.6*以上。

## 解决依赖
* Python >= 3.6
* NumPy
* SciPy


## 特色

* 采用了和*scikit-learn*类似的接口，如`fit`和`predict`方法。
* 使用*NumPy*加速。
* 使用了变量注解。

## 快速入门
Input:
```python
from plume.utils import plot_decision_boundary

X, y = sklearn.datasets.make_moons(200, noise=0.20)
y = 2 * y - 1
clf = SVC(C=3, kernel='rbf')
clf.fit(X, y)
plot_decision_boundary(clf.predict, X, y, 'Support Vector Machine')
```
Output:

![img](docs/figure_svm.png)