from skimage.color import hsv2rgb
import numpy as np
import cv2

def ra():
    return hsv2rgb([np.random.rand(), 1, 1])
_lh = 0
def scol():
    global _lh
    c = hsv2rgb([_lh, 1, 1])
    _lh += 0.15
    return c

def bbxywh(bb):
    return (bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0]
def bbctr(bb):
    xy, w, h = bbxywh(bb)
    return (xy[0]+w/2, xy[1]+h/2)
def bb_aspect(bb):
    _, w, h = bbxywh(bb)
    return w/h
def bb_ar(bb):
    _, w, h = bbxywh(bb)
    return w*h

def add_text(txt, xy, col=(255, 255, 255), img=None, ax=None, font=cv2.FONT_HERSHEY_SIMPLEX, bg=True):
    x, y = int(xy[0]), int(xy[1])
    # if bg:
    #     w = len(txt)*40
    #     h = 20
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,0), -1)
    cv2.putText(img, txt, (x, y), font, 1, (0,0,0), 5)
    cv2.putText(img, txt, (x, y), font, 1, col, 2)
    pass

def regbound(r, img, ofs=[0, 0], col=None):
    bb = r.bbox
    x, w, h = bbxywh(bb)
    if col is None: col = ra()
    ix, iy = ofs[0]+x[0], ofs[1]+x[1]
    x1, y1 = ix+w, iy+h
    cv2.rectangle(img,(ix,iy),(x1,y1), col,3)
    return col
def touches_border(b, bor, sl=1):
    return b[0] <= bor[0]+sl or b[1] <= bor[1]+sl or b[2] >= bor[2]-sl or b[3] >= bor[3]-sl