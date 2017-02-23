import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from mlearn.randomforests import RandomForestsClassifier
import numpy as np


def main():
    train_x = [
        ['有钱', '有房', '长得丑'],
        ['没钱', '有房', '长得丑'],
        ['有钱', '没房', '长得丑'],
        ['没钱', '没房', '长得丑'],
        ['没钱', '没房', '长得帅'],
    ]
    train_x = np.array(train_x)
    train_y = np.array(['嫁', '嫁', '嫁', '不嫁', '不嫁'])
    x = train_x[4]
    clf = RandomForestsClassifier()
    clf.fit(train_x, train_y)
    print(clf.predict_one(x))


if __name__ == '__main__':
    main()