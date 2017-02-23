import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from mlearn.naive_bayes import NaiveBayesClassifier
import numpy as np


def main():
    train_x = [["1", "S"], ["1", "M"], ["1", "M"],
               ["1", "S"], ["1", "S"], ["2", "S"],
               ["2", "M"], ["2", "M"], ["2", "L"],
               ["2", "L"], ["3", "L"], ["3", "M"],
               ["3", "M"], ["3", "L"], ["3", "L"]]
    train_y = ["-1", "-1", "1", "1", "-1",
               "-1", "-1", "1", "1", "1",
               "1", "1", "1", "1", "-1"]

    n = NaiveBayesClassifier()
    n.fit(train_x, train_y)
    ans = n.predict([["2", "S"]])
    print(ans)

if __name__ == '__main__':
    main()