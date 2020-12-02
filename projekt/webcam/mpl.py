import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    return plt.scatter(rand_jitter(x), rand_jitter(y), s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)


a = np.array([20, 10, 19, 10, 19, 10, 13, 20, 16, 20, 12, 19, 12, 20, 16, 20, 15, 18, 12, 19, 12, 19, 15, 19, 12, 19, 15, 17]).reshape((-1, 2))

ncs = list(range(1, 7))
iner = []
for cc in ncs:
    k = KMeans(n_clusters=cc).fit(a)
    iner.append(k.inertia_)
iner = np.array(iner)
print(np.where(iner < 30)[0][0])

dif = np.diff(np.diff(iner))
nc = np.argmax(dif)+2

plt.plot(ncs[:-2], dif)
plt.plot(ncs, iner)

km = KMeans(n_clusters=nc).fit(a)
kml = km.labels_

k = KMeans(n_clusters=3).fit(a)

print(k.labels_)


#jitter(a[:,0], a[:,1])
plt.show()