from .math_module import xp, xcipy, ensure_np_array
from lina import utils

import numpy as np
from IPython.display import clear_output
import skimage

from matplotlib.patches import Circle
from matplotlib.colors import LogNorm

def measure_waffle_center_and_angle(
        waffle_im, 
        psf_pixelscale_lamD, 
        im_thresh=1e-4, 
        r_thresh_min=12,
        r_thresh_max=18, 
        xc=0,
        yc=0,
        verbose=True, 
        plot=True,
    ):
    npsf = waffle_im.shape[0]
    y,x = (xp.indices((npsf, npsf)) - npsf//2)*psf_pixelscale_lamD
    r = xp.sqrt((x - xc)**2 + (y - yc)**2)
    waffle_mask = (waffle_im > im_thresh) * (r>r_thresh_min) * (r<r_thresh_max)

    centroids = []
    for i in [0,1]:
        for j in [0,1]:
            arr = waffle_im[j*npsf//2:(j+1)*npsf//2, i*npsf//2:(i+1)*npsf//2]
            mask = waffle_mask[j*npsf//2:(j+1)*npsf//2, i*npsf//2:(i+1)*npsf//2]
            cent = np.flip(skimage.measure.centroid(ensure_np_array(mask*arr)))
            cent[0] += i*npsf//2
            cent[1] += j*npsf//2
            centroids.append(cent)

    centroids.append(centroids[0])
    centroids = np.array(centroids)
    centroids[[2,3]] = centroids[[3,2]]
    if verbose: print('Centroids:\n', centroids)

    mean_angle = 0.0
    for i in range(4):
        angle = np.arctan2(centroids[i+1][1] - centroids[i][1], centroids[i+1][0] - centroids[i][0]) * 180/np.pi
        if angle<0:
            angle += 360
        if 0<angle<90:
            angle = 90-angle
        elif 90<angle<180:
            angle = 180-angle
        elif 180<angle<270:
            angle = 270-angle
        elif 270<angle<360:
            angle = 360-angle
        mean_angle += angle/4
    if verbose: print('Angle: ', mean_angle)

    m1 = (centroids[0][1] - centroids[2][1])/(centroids[0][0] - centroids[2][0])
    m2 = (centroids[1][1] - centroids[3][1])/(centroids[1][0] - centroids[3][0])
    # print(m1,m2)
    b1 = -m1*centroids[0][0] + centroids[0][1]
    b2 =  -m2*centroids[1][0] + centroids[1][1]
    # print(b1,b2)

    # m1*x + b1 = m2*x + b2
    # (m1-m2) * x = b2 - b1
    xc = (b2 - b1) / (m1 - m2)
    yc = m1*xc + b1
    print('Measured center in X: ', xc)
    print('Measured center in Y: ', yc)

    xshift = -np.round(npsf/2 - xc)
    yshift = -np.round(npsf/2 - yc)
    print('Required shift in X: ', xshift)
    print('Required shift in Y: ', yshift)

    if plot: 
        patches1, patches2 = [], []
        for i in range(4): 
            patches1.append(Circle(centroids[i]-npsf//2, 1, fill=False, color='black'))
            patches2.append(Circle(centroids[i]-npsf//2, 1, fill=False, color='black'))
        patches1.append(Circle((xshift,yshift), 3, fill=True, color='red'))
        patches2.append(Circle((xshift,yshift), 3, fill=True, color='red'))
        utils.imshow(
            [waffle_im, waffle_mask, waffle_mask*waffle_im], 
            norms=[LogNorm(np.max(waffle_im)/1e4), LogNorm(np.max(waffle_im)/1e4), LogNorm(np.max(waffle_im)/1e4)],
            pxscls=3*[1],
            grids=3*[1],
            all_patches=[patches1, patches2],
        )
    
    return xshift, yshift, mean_angle


