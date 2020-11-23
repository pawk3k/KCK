#!/usr/bin/env python
# coding: utf-8

# In[1]:

from skimage import io
from skimage.color import rgb2gray
from matplotlib import pyplot as plt, colors
from glob import glob
import numpy as np
from itertools import product
import skimage.morphology as mp
from skimage import measure, feature
from skimage.measure import find_contours as fc


# In[2]:


planes_orig = [io.imread(fn) for fn in sorted(glob("data/samolot??.jpg"))]


# In[3]:


### Helper functions for grouping contours belonging to the same plane

# returns vertices of a box around a list of points
def bbox(c):
    diag = *np.min(c, 0), *np.max(c, 0)
    return list(product(diag[::2], diag[1::2]))

def bboxwh(bb):
    return bb[3][0]-bb[0][0], bb[3][1]-bb[0][1]

def in_bbox(p, diag, sl=0):
    return diag[0][0]-sl < p[0] < diag[1][0]+sl and             diag[0][1]-sl < p[1] < diag[1][1]+sl

# approximate bounding box as circle
def bboxr(bb):
    return min(bboxwh(bb))/2
def bboxcr(bb):
    r = bboxr(bb)
    ctr = ((bb[0][0]+bb[3][0])/2, (bb[0][1]+bb[3][1])/2)
    return ctr, r

# slack as percent 0-1
def circ_int(bb1, bb2, slp=0):
    c1, r1 = bboxcr(bb1)
    c2, r2 = bboxcr(bb2)
    sl = max(r1, r2)*slp
    return (c2[0]-c1[0])**2+(c2[1]-c1[1])**2 < (r1+r2+sl)**2

def intersect(bb1, bb2, slack=0):
    d1 = [bb1[0], bb1[3]]
    d2 = [bb2[0], bb2[3]]
    return any(in_bbox(p, d2, slack) for p in bb1) or             any(in_bbox(p, d1, slack) for p in bb2)


# In[4]:


from random import random

def draw_contour(image, ax=plt):
    
    pp = rgb2gray(image)
    
    # special case if brightness doesn't deviate much
    if pp.std() <= 0.05:
        MIN = pp.mean() - pp.std()
        MAX = pp.mean() + pp.std()
        norm = (pp - MIN) / (MAX - MIN)
        norm[norm > 1] = 1
        norm[norm < 0] = 0
        pp = norm
    
    # remove details of clouds
    pp = pp.clip(0, np.percentile(pp, 70))

    # enhance contours
    pp = feature.canny(pp, sigma=3)
    pp = (pp > 0.7)*1.0
    pp = mp.dilation(pp)  
    
    
    # find contours and order descending by size (ungrouped)
    cs = fc(pp, level=0.8)
    cs.sort(key=lambda x:bboxr(bbox(x)), reverse=True)
    
    # group contours
    cs2 = []
    bboxes = []
    slack = 0.4
    for j, c in enumerate(cs):
        bb = bbox(c)
            
        for i, (c1, bb1) in enumerate(zip(cs2, bboxes)):
            if circ_int(bb, bb1, slack) or intersect(bb, bb1, 0):
                c1 = cs2[i] = np.append(c1, c, 0)
                bboxes[i] = bbox(cs2[i])
                break
        else:
            cs2.append(c)
            bboxes.append(bbox(c))
    
    # draw contours
    last_hue = random()
    for c in cs2:
        # centroid approximation
        ctr = np.mean(c, 0)
        
        hsv = (last_hue, 1, 1)
        last_hue = (last_hue+0.15)%1
        col = [[*colors.hsv_to_rgb(hsv)]]
        ax.scatter(c[:,1], c[:,0], [0.5]*len(c), c=col)
        ax.scatter([ctr[1]], [ctr[0]], c='w')


# In[5]:


def collage(images, ncols=3, width=10, imparams={}, nplanes=None):
    sh = images[0].shape
    aspect = sh[0] / sh[1]
    
    if nplanes is None: nplanes = len(images)
    nrows = int(np.ceil(nplanes/ncols))
    
    fig, axes = plt.subplots(nrows, ncols,
                             gridspec_kw={"wspace": 0, "hspace": 0})
    fig.set_size_inches(width, width*aspect*nrows/ncols)
    axes = axes.flatten()
    
    
    
    for i in range(nplanes):
        ax, img = axes[i], images[i]
        print("{}/{}".format(i+1, nplanes))
        ax.axis("off")
        ax.imshow(img, aspect='auto')
        draw_contour(img, ax)
    plt.show()


# In[6]:


collage(planes_orig)

