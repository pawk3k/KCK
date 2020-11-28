import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from skimage import io, draw
import skimage.color as sc
from skimage.color import rgb2gray as r2g, rgb2hsv as r2h
from skimage.feature import canny
from skimage.morphology import dilation, erosion
from skimage.filters import gaussian
from ipywidgets import interact
from skimage.morphology import convex_hull_image as conhu
from skimage.measure import label, regionprops

def sh(image, ax=plt, **kwargs):
    ax.imshow(image, cmap='gray', **kwargs)
def colax(m, n):
    fig, axs = plt.subplots(m, n)#, gridspec_kw={"wspace": 0, "hspace": 0})
    return fig, (a for a in axs.flatten())
def ra():
    return sc.hsv2rgb([np.random.rand(), 1, 1])
_lh = 0
def scol():
    global _lh
    c = sc.hsv2rgb([_lh, 1, 1])
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
def regbound(r, ax, ofs=[0, 0], col=None):
    bb = r.bbox
    x, w, h = bbxywh(bb)
    if col is None: col = ra()
    rect = Rectangle((ofs[0]+x[0], ofs[1]+x[1]), 
                        w, h, fill=None, lw=2, ec=col)
    ax.add_patch(rect)
    return col
