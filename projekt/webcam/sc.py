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

    #rp = regionprops(label(res, background=0))

    rsub = np.sqrt(iw*ih)

    r0 = int(rsub*0.1)
    r1 = int(rsub*0.2)
    rx = np.arange(r0, r1)




    # min_a = 0.0015*iw*ih
    # max_a = 0.05*iw*ih
    # rp2 = list(filter(lambda r1: not touches_border(r1.bbox, (0, 0, ih, iw)) 
    #                     and min_a < bb_ar(r1.bbox) < max_a , rp))

    for r in rp:
        regbound(r, img, ofs=ofs, col=(0, 0, 255), th=1)




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