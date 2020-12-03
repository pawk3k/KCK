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
from skimage import io

from grouping import group_dots

import matplotlib.pyplot as plt

BOUNDS = True

font = cv2.FONT_HERSHEY_SIMPLEX

def label_cluster(img, x, cx, cy, lbl, fs=25, col=(255,255,255)):
    #ax.text(cx+x[0]-fs/2, cy+x[1], lbl, color=col, fontsize=fs, fontweight='bold')
    add_text(lbl, (cx+x[0]-fs/2, cy+x[1]), col, img)

IMG_IX = 0
MODE = 0

def mark_dots(sub, img, ofs=(0, 0), group=False):
    global IMG_IX, _lh, MODE
    iw, ih = sub.shape
    bl = sub # gau(sub, 4)
    res = canny(bl, sigma=1)

    res = closing(res)

    ovl = np.repeat(adim(res), 3, -1)

    #img[ofs[1]:ofs[1]+iw, ofs[0]:ofs[0]+ih] = ovl

    rp = regionprops(label(res, background=0))

    rsub = np.sqrt(iw*ih)/2

    # r0 = int(rsub*0.12)
    # r1 = int(rsub*0.14)
    # rx = np.arange(r0, r1)
    # hcr = hc(res, rx)

    # cth = 0.5
    # hcr[hcr < cth] = 0

    # _, cx, cy, ri = hcp(hcr, rx, 5, 5, 0.37, total_num_peaks=10)

    # #print("CNT", len(ri))

    # for x, y, r in zip(cx, cy, ri):
    #     cv2.circle(img, (x+ofs[0],y+ofs[1]), r, (0, 255, 0), 2)

    ar = iw*ih
    min_a = 0.0015*ar
    max_a = 0.04*ar

    def size_ok(bb):
        _, w, h = bbxywh(bb)
        ar = w*h
        asp = w/h
    
    def is_black(r):
        global IMG_IX
        x, w, h = bbxywh(r.bbox)
        if w < 2 or h < 2:
            return 0
        nbb = bbxywh(sanit(*bbexp(r.bbox, 5), iw, ih))
        nx, nw, nh = nbb
        cr1 = crop1(res, x, w, h)
        cr = crop1(res, *nbb)
        crbw = crop1(sub, *nbb)
        mask1 = conhu(cr1)
        mask = cr

        dx = (x[0]-nx[0], x[1] - nx[1])

        mask[dx[1]:dx[1]+h, dx[0]:dx[0]+w] = mask1
        
        # print((x, w, h), nbb)

        # io.imsave("sub/cr.png", np.uint8(255*cr))
        # io.imsave("sub/cr1.png", np.uint8(255*mask))
        # input()
        # IMG_IX += 1
        

        try:
            cb = np.average(crbw, weights=mask)
            cw = np.average(crbw, weights=1-mask)
        except Exception:
            #print("FALSE")
            return False
        #print(cb, cw)
        return cw-cb > 0.1

    def is_dot(r):
        _, w, h = bbxywh(r.bbox)
        if w*h > 0.3*ar or w*h < 0.005*ar:
            return False
        if MODE == 0 and group == False:
            max_asp = 1.2
            if not (1/max_asp < w/h < max_asp):
                return False
        return not touches_border(r.bbox, (0, 0, ih, iw)) and is_black(r)

    # for r in rp:
    #     regbound(r, img, ofs=ofs, col=(255, 255, 0), th=1)

    rp = list(filter(is_dot, rp))

    for r in rp:
        regbound(r, img, ofs=ofs, col=(255, 255, 0), th=1)

    rp = list(filter(lambda x: not in_sth(x, rp), rp))

    regrouped = False
    # GROUP DOTS
    if group:
        regrouped = group_dots(rp, img, ofs=ofs)
    if not regrouped and len(rp) > 0:
        for r in rp:
            regbound(r, img, ofs=ofs, col=(0, 0, 255), th=2)
        add_text(str(len(rp)), (ofs[0]+iw/2, ofs[1]+ih/2), img=img)

    # return number of dots
    return len(rp)


def transform2(im, group=False):
    bw = r2g(im)
    iw, ih = bw.shape
    
    bl = gau(bw, 3)
    c = canny(bl, 4)
    c = np.uint8(255*c)

    la = label(c, background=0)
    rp = regionprops(la)

    img = im.copy() #sc.gray2rgb(bw)

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

    
    rp2 = []

    for r in rp:
        if not in_sth(r, rp):
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

        #bbbound(nbb, img, col=(255,0,2525))
        sub = bw[nx[1]:nx[1]+nh, nx[0]:nx[0]+nw]

        ndots = mark_dots(sub, img, ofs=nx, group=group)

        #img[x[1]:x[1]+h,x[0]:x[0]+w] = mc
        #img[nx[1]:nx[1]+nh,nx[0]:nx[0]+nw] = mc
        if BOUNDS and ndots > 0:
            #regbound(r, img, col=(255, 0, 255))
            bbbound((nx[1], nx[0], nx[1]+nh, nx[0]+nw), img, col=(255, 0, 255))
    #img = sc.gray2rgb(bw)
    #ii =  np.uint8(255*img)
    #print(ii.shape)
    return img

def showImage(fname="im2.png", group=True):
    cv2.namedWindow("preview")

    frame = io.imread(fname)
    tr = transform2(frame, group=group)
    cv2.imshow("preview", tr)
    while True:
        key = cv2.waitKey(20)
        if key == 114:
            d = fname.rfind(".")
            fname = fname[:d]+"_res1"+fname[d:]
            io.imsave(fname, cv2.cvtColor(tr, cv2.COLOR_BGR2RGB))
            print("SAVED TO", fname)
        elif key == 27:
            break
        elif key == 105:
            BOUNDS = not BOUNDS

    cv2.destroyAllWindows()

import os
def next_index(fmt):
    i = 1
    while os.path.isfile(fmt % i):
        i += 1
    return i

def showStream(group=False, mode=1):
    global BOUNDS, MODE
    cv2.namedWindow("preview")
    #cv2.namedWindow("bw1")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        print(frame.shape)
    else:
        rval = False

    while rval:
        
        tr = transform2(frame, group=group)
        #tr, ex = process(frame)
        cv2.imshow("preview", tr)
        #cv2.imshow("bw1", ex)

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
        if key == 112:
            frame, tr = [cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB) for im_cv in [frame, tr]]
            fmt = "res/im%03d.png"
            ix = next_index(fmt)
            io.imsave(fmt % ix, frame)
            io.imsave("res/im%03d_res.png" % ix, tr)
        elif key == 105:
            BOUNDS = not BOUNDS
        else:
            pass#print(key)

        rval, frame = vc.read()
        
        
    cv2.destroyWindow("preview")

if __name__=="__main__":
    showStream(group=False, mode=0)
    #showImage("res/im002.png", group=True)
    