from plume.cluster import KMeans
import numpy as np


def test_kmeans():
    clf = KMeans(3)
    X = np.array([
        [1, 1], [0.9, 1.2], [1.3, 0.8],
        [8, 18], [8.1, 17.9], [8.2, 17.8],
        [19, 0], [19, 0.2], [19.2, 0.3]
    ])

    clf.fit(X)
    print(clf.predict(X))


if __name__ == '__main__':
    test_kmeans()
