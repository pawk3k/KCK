
import numpy as np
from sklearn.cluster import KMeans
from utils import *

def group_dots(rp, img, ofs=(0,0)):
    px = np.array([bbwh(r.bbox) for r in rp])

    #plt.scatter(px[:0], px[:1])

    le = len(px)
    if le < 2:
        return False
    else:
        coef = KMeans(n_clusters=1).fit(px).inertia_
        co_th = 150/le
        if coef < co_th:
            return False
        else:
            ncs = list(range(1, min(le, 4)))
            iner = []
            for cc in ncs:
                k = KMeans(n_clusters=cc).fit(px)
                iner.append(k.inertia_)
            
            iner = np.array(iner)

            if iner[-1] >= co_th:
                nc = ncs[-1]
            else:
                k = np.where(iner < co_th)[0][0]
                nc = k+1
            # dif = np.diff(np.diff(iner))
            # nc = np.argmax(dif)+2

            # print(iner)

            km = KMeans(n_clusters=nc).fit(px)
            kml = km.labels_

            _lh = 0.5
            cols = [255*scol() for _ in range(nc)]


            sets = [[rp[i] for i in range(len(kml)) if kml[i] == j] for j in range(nc)]
            for si, s in enumerate(sets):
                col = cols[si]
                mid = np.mean([bbctr(r.bbox) for r in s], axis=0)
                cx, cy = mid
                for reg in s:
                    regbound(reg, img, ofs=ofs, col=col, th=2)
                txt = str(len(s))
                add_text(txt, (ofs[0]+cy, ofs[1]+cx), col=col, img=img, bdr_col=255-col)
            return True