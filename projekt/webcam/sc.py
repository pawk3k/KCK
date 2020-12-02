import cv2
import numpy as np
from skimage.color import rgb2gray as r2g, rgb2hsv as r2h
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
from skimage.morphology import convex_hull_image as conhu

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

    # cond0 = np.logical_and(im[:,:,0] > thw, im[:,:,1] > thw, im[:,:,2] > thw)

    q = bw1.copy()

    q = gau(q, 3)

    q[q < thw] = 0
    q[cond0] = 1
    

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

    

    for r in rp:
        x, w, h = bbxywh(r.bbox)
        ar = w*h
        #if not (wh0 < w < wh1 and wh0 < h < wh1):
        if ar < 30 or w > wh1 or h > wh1:
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


def mark_dots(sub, img, ofs=(0, 0)):
    iw, ih = sub.shape
    # try:
    #     th = thm(sub)
    # except RuntimeError:
    #     return
    # sub = np.uint8(255*(sub > th))
    #vth = 0.2
    #sub[sub < vth] = 0
    #sub[sub >= vth] = 1
    bl = sub # gau(sub, 4)
    res = canny(bl, sigma=0.5)

    ovl = np.repeat(adim(res), 3, -1)

    img[ofs[1]:ofs[1]+iw, ofs[0]:ofs[0]+ih] = ovl

    rp = regionprops(label(sub, background=255))

    # min_a = 0.0015*iw*ih
    # max_a = 0.05*iw*ih
    # rp2 = list(filter(lambda r1: not touches_border(r1.bbox, (0, 0, ih, iw)) 
    #                     and min_a < bb_ar(r1.bbox) < max_a , rp))

    # for r in rp2:
    #     regbound(r, img, ofs=ofs)




def transform2(im):
    bw = r2g(im)
    iw, ih = bw.shape
    
    bl = gau(bw, 3)
    c = canny(bl, 4)
    c = np.uint8(255*c)

    la = label(c, background=0)
    rp = regionprops(la)

    img = sc.gray2rgb(bw)

    def is_white(r):
        x, w, h = bbxywh(r.bbox)
        
        cr = crop1(im, x, w, h)
        subi = crop1(c, x, w, h)
        subbw = crop1(bw, x, w, h)
        mask = conhu(subi)
        mask[subbw < 0.3] = 0
        wei = mask.flatten()
        
        try:
            mc = np.average(cr.reshape((-1, 3)), 0, wei)/255
        except ZeroDivisionError:
            return False
        
        hsv = r2h(mc)
        return hsv[1] < 0.4 and hsv[2] > 0.7

    rp = filter(lambda r: is_white(r) and bb_ar(r.bbox) < iw*ih/8, rp)
    rp = list(rp)

    def in_sth(r):
        for r1 in rp:
            if r1 is not r:
                if rinr(r, r1):
                    return True
        return False
    rp2 = []

    for r in rp:
        if not in_sth(r):
            rp2.append(r)

    rp = rp2

    for r in rp:
        x, w, h = bbxywh(r.bbox)

        if w < h:
            dw = (h-w)/2
            dh = 0
        else:
            dh = (w-h)/2
            dw = 0
        na = max(w, h)

        nbb = bbexp2(r.bbox, dh+na/8, dw+na/8)
        nbb = sanit(*nbb, iw, ih)
        nx, nw, nh = bbxywh(nbb)

        bbbound(nbb, img, col=(255,0,2525))
        sub = bw[nx[1]:nx[1]+nh, nx[0]:nx[0]+nw]

        mark_dots(sub, img, ofs=nx)

        #img[x[1]:x[1]+h,x[0]:x[0]+w] = mc
        #img[nx[1]:nx[1]+nh,nx[0]:nx[0]+nw] = mc
        # regbound(r, img, col=(255, 0, 255))
    #img = sc.gray2rgb(bw)
    return img

if __name__=="__main__":
    cv2.namedWindow("preview")
    #cv2.namedWindow("bw1")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        print(frame.shape)
    else:
        rval = False

    while rval:
        
        tr = transform2(frame)
        #tr, ex = process(frame)
        cv2.imshow("preview", tr)
        #cv2.imshow("bw1", ex)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")