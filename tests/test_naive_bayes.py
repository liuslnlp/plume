from plume.naive_bayes import GaussianNB, MultinomialNB
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB as SKGaussianNB
import numpy as np

def test_gaussiannb():
    iris = load_iris()
    clf = GaussianNB()
    clf.fit(iris.data, iris.target)
    y_pred = clf.predict(iris.data)
    print(y_pred)
    clf_ = SKGaussianNB()
    clf_.fit(iris.data, iris.target)
    print(clf_.predict(iris.data))


    print(iris.target)

def test_multinomialnb():
    X = np.random.randint(5, size=(6, 100))
    y = np.array([1, 2, 3, 4, 5, 6])
    clf = MultinomialNB()
    clf.fit(X, y)
    print(clf.predict(X[2:3]))


if __name__ == '__main__':
    test_multinomialnb()