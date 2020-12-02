from sklearn.cluster import KMeans
import numpy as np

#a = np.array([0, 1, 0, 2, 50, 0, 50, 1, 100, 0, 100, 1]).reshape((-1, 2))
a = np.array([0, 1, 0, 5, 0, 9, 100, 1, 100, 5, 100, 9]).reshape((-1, 2))
print(a)
km = KMeans(n_clusters=2).fit(a)

print(km.inertia_)