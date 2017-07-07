from plume.perceptron import PerceptronClassifier
import numpy as np

x_train = np.array([[3, 3], [4, 3], [1, 1]])
y_train = np.array([1, 1, -1])

clf = PerceptronClassifier(dual=False)

clf.fit(x_train, y_train)
print(clf.get_model())
print(clf.predict(x_train))

clf1 = PerceptronClassifier()

clf1.fit(x_train, y_train)
print(clf1.get_model())
print(clf1.predict(x_train))
