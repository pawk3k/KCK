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

def adim(x):
    return np.expand_dims(x, -1)

# matplotlib
def bbxywh(bb):
    return (bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0]


def bbxywh2(bb):
    return (bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1]

def bbctr(bb):
    xy, w, h = bbxywh2(bb)
    return [xy[0]+w/2, xy[1]+h/2]
def bb_aspect(bb):
    _, w, h = bbxywh(bb)
    return w/h
def bb_ar(bb):
    _, w, h = bbxywh(bb)
    return w*h
def bb_dia(bb):
    _, w, h = bbxywh(bb)
    return np.sqrt(w*h)

def bbwh(bb):
    _, w, h = bbxywh(bb)
    return [w, h]


def bbinbb(bb1, bb2):
    isin = bb1[0] >= bb2[0] and bb1[2] <= bb2[2] and bb1[1] >= bb2[1] and bb1[3] <= bb2[3]
    return isin

def rinr(r1, r2):
    return bbinbb(r1.bbox, r2.bbox)

def pinbb(p, bb):
    return bb[0] < p[0] < bb[2] and bb[1] < p[1] < bb[3]

def rinr2(r1, r2):
    ctr = bbctr(r1.bbox)
    isin = pinbb(ctr, r2.bbox)
    return isin

def add_text(txt, xy, col=(255, 255, 255), img=None, ax=None, font=cv2.FONT_HERSHEY_SIMPLEX, bdr_col=(0, 0, 0)):
    x, y = int(xy[0]), int(xy[1])
    # if bg:
    #     w = len(txt)*40
    #     h = 20
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,0), -1)
    cv2.putText(img, txt, (x, y), font, 1, bdr_col, 5)
    cv2.putText(img, txt, (x, y), font, 1, col, 2)
    pass

def bbbound(bb, img, ofs=[0, 0], col=None, th=3):
    x, w, h = bbxywh(bb)
    if col is None: col = ra()
    ix, iy = ofs[0]+x[0], ofs[1]+x[1]
    x1, y1 = ix+w, iy+h
    cv2.rectangle(img,(ix,iy),(x1,y1), col,th)
    return col

def regbound(r, img, ofs=[0, 0], col=None, th=3):
    bb = r.bbox
    return bbbound(bb, img, ofs=ofs, col=col, th=th)


def touches_border(b, bor, sl=1):
    return b[0] <= bor[0]+sl or b[1] <= bor[1]+sl or b[2] >= bor[2]-sl or b[3] >= bor[3]-sl


def crop1(img, xy, w, h):
        return img[xy[1]:xy[1]+h, xy[0]:xy[0]+w]

def tcrop1(img, xy, w, h):
    return crop1(img, (xy[1], xy[0]), h, w)

def crop2(img, x0, y0, x1, y1):
    w, h = img.shape[:2]
    x0 = min(max(0, x0), w)
    x1 = min(max(0, x1), w)
    y0 = min(max(0, y0), h)
    y1 = min(max(0, y1), h)
    return img[x0:x1, y0:y1]

def bbexp(bb, v):
    return [bb[0]-v, bb[1]-v, bb[2]+v, bb[3]+v]

def bbexp2(bb, dw, dh):
    return [bb[0]-dw, bb[1]-dh, bb[2]+dw, bb[3]+dh]

def sanit(x0, y0, x1, y1, w, h):
    x0 = int(min(max(0, x0), w))
    x1 = int(min(max(0, x1), w))
    y0 = int(min(max(0, y0), h))
    y1 = int(min(max(0, y1), h))
    return x0, y0, x1, y1

def contr(img, bb, v):
    i1 = crop2(img, *bb)
    i2 = crop2(img, *bbexp(bb, v))
    mask = np.zeros_like(i2)
    mask[v:-v, v:-v] = 1
    mask = 1-mask
    m1 = np.mean(i1)
    m2 = np.average(i2, weights=mask)

    df = m2 - m1
    return df

def in_sth(r, rp):
        for r1 in rp:
            if r1 is not r:
                if rinr2(r, r1) and bb_ar(r.bbox) < bb_ar(r1.bbox):
                    return True
        return False