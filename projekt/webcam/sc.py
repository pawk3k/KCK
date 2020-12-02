import cv2
import numpy as np
from skimage.color import rgb2gray as r2g
import skimage.color as sc
from skimage.feature import canny
from skimage.filters import gaussian as gau
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from sklearn.cluster import KMeans
from skimage.morphology import erosion, dilation, area_closing, closing, square
from skimage.transform import hough_circle as hc, hough_circle_peaks as hcp
from utils import *
from numpy import unravel_index as urix
from skimage.measure import find_contours as fc
from skimage.measure import label, regionprops
from skimage.filters import threshold_minimum as thm

font = cv2.FONT_HERSHEY_SIMPLEX

def label_cluster(img, x, cx, cy, lbl, fs=25, col=(255,255,255)):
    #ax.text(cx+x[0]-fs/2, cy+x[1], lbl, color=col, fontsize=fs, fontweight='bold')
    add_text(lbl, (cx+x[0]-fs/2, cy+x[1]), col, img)




def process(imbase, group_dots=True, label_pics=True):
    global _lh
    im2 = imbase.copy()
    im3 = r2g(im2)
    #th = (np.percentile(im3, 10) + np.percentile(im3, 90))/2 + 0.1 #0.5
    #th = thm(im3)
    #im3 = np.uint8(255*(im3 > th))
    th = 0.5
    im3[im3 < th] = 0
    im3[im3 >= th] = 1

    iw, ih = im3.shape

    img = im2#sc.gray2rgb(im3)
    txtcol=(255, 255, 255)
    
    if not label_pics:
        return
    ls = label(im3, background=0)
    rp = regionprops(ls)
    mears = [r.area for r in rp]
    mn = np.mean(mears)
    rp = list(filter(lambda r: r.area>0.1*mn, rp))

    for i, r in enumerate(rp):
        x, w, h = bbxywh(r.bbox)

        if touches_border(r.bbox, (0, 0, iw, ih)): continue

        col = regbound(r, img)
        #cv2.putText(img, str(i+1), (x[0]+w/2, x[1]-20), font, 1, col, 2)
        subi = im3[x[1]:x[1]+w, x[0]:x[0]+h]
        
        rp2 = regionprops(label(subi, background=255))
        min_a = 0.0015*w*h
        max_a = 0.05*w*h
        rp2 = list(filter(lambda r1: not touches_border(r1.bbox, (0, 0, h, w)) 
                          and min_a < bb_ar(r1.bbox) < max_a , rp2))
        px = np.array([bbctr(r.bbox) for r in rp2])

        if len(px) < 1:
            continue
        if group_dots:
            coef = KMeans(n_clusters=1).fit(px).inertia_/np.sqrt(iw*ih)
        if group_dots and coef > 10:
            ncs = list(range(1, len(rp2)))
            iner = []
            for cc in ncs:
                k = KMeans(n_clusters=cc).fit(px)
                iner.append(k.inertia_)
            
            dif = np.diff(np.diff(iner))
            nc = np.argmax(dif)+2 if len(dif) > 0 else 1
            
            km = KMeans(n_clusters=nc).fit(px)
            _lh = 0.5
            cols = [255*scol() for _ in range(nc)]
            for j, r2 in enumerate(rp2):
                regbound(r2, img, ofs=x, col=cols[km.labels_[j]])
            for ci in range(nc):
                cx, cy = km.cluster_centers_[ci]
                t = str(len(km.labels_[km.labels_==ci]))
                label_cluster(img, x, cx, cy, t, col=txtcol)
        else:
            col1 = 255*scol()
            for r2 in rp2:
                regbound(r2, img, ofs=x, col=col1)
            label_cluster(img, x, w/2, h/2, str(len(rp2)), col=txtcol)
    return img, im3

def transform(im):
    bw = r2g(im)
    iw, ih = bw.shape
    bw1 = bw.copy()
    #bl = gau(bw, 3)
    
    #bw = gau(bw, 4)

    thw = np.percentile(bw1, 95) * 0.8

    q = bw1.copy()

    q = gau(q, 3)

    q[q<thw] = 0
    q[q >= thw] = 1

    th = 20
    
    cond = np.logical_and(im[:,:,0] < th, im[:,:,1] < th, im[:,:,2] < th)

    bw[cond] = 0
    bw[np.logical_not(cond)] = 1

    #bl = #gau(bw, 1)

    c = canny(bw, sigma=2)
    c = np.uint8(c*255)

    #c = erosion(dilation(c))

    crgb = sc.gray2rgb(c)


    lab = label(bw, background=1)

    rp = regionprops(lab)
    _lh = 0
    wh0, wh1 = 3, 20

    def crop1(img, xy, w, h):
        return img[xy[1]:xy[1]+h, xy[0]:xy[0]+w]

    def crop2(img, x0, y0, x1, y1):
        w, h = img.shape[:2]
        x0 = min(max(0, x0), w)
        x1 = min(max(0, x1), w)
        y0 = min(max(0, y0), h)
        y1 = min(max(0, y1), h)
        return img[x0:x1, y0:y1]

    def bbexp(bb, v):
        return [bb[0]-v, bb[1]-v, bb[2]+v, bb[3]+v]

    def contr(bb, v):
        i1 = crop2(bw1, *bb)
        i2 = crop2(bw1, *bbexp(bb, v))
        mask = np.zeros_like(i2)
        mask[v:-v, v:-v] = 1
        mask = 1-mask
        m1 = np.mean(i1)
        m2 = np.average(i2, weights=mask)

        df = m2 - m1
        #print(m1, m2, df)
        return df

    for r in rp:
        x, w, h = bbxywh(r.bbox)
        ar = w*h
        #if not (wh0 < w < wh1 and wh0 < h < wh1):
        if ar < 40 or w > wh1 or h > wh1:
            #crgb[x[1]:x[1]+h,x[0]:x[0]+w] = [0,0,0]
            continue
        regbound(r, crgb, col=255*scol())

    # rx = np.arange(3, 10)
    # hcr = hc(c, rx)

    # cth = 0.5
    # hcr[hcr < cth] = 0

    # _, cx, cy, ri = hcp(hcr, rx, 5, 5, 0.5, total_num_peaks=20)

    # for x, y, r in zip(cx, cy, ri):
    #     cv2.circle(crgb, (x,y), r, (255, 0, 0), 4)

    #c = gau(c, 5)

    return crgb, q#np.uint8(c*255)


if __name__=="__main__":
    cv2.namedWindow("preview")
    cv2.namedWindow("bw1")
    vc = cv2.VideoCapture(2)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        print(frame.shape)
    else:
        rval = False

    while rval:
        
        tr, ex = transform(frame)
        #tr, ex = process(frame)
        cv2.imshow("preview", tr)
        cv2.imshow("bw1", ex)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")