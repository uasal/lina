from .math_module import xp, _scipy, cupy_avail
if cupy_avail:
    import cupy as cp
else:
    cp = False

from . import utils

import numpy as np
import scipy

import astropy.units as u

import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize
from IPython.display import display, clear_output

def imshow1(arr, 
            title=None, 
            xlabel=None,
            npix=None,
            lognorm=False, vmin=None, vmax=None,
            cmap='magma',
            pxscl=None,
            axlims=None,
            patches=None,
            grid=False, 
            figsize=(4,4), dpi=125, 
            display_fig=True, return_fig=False):
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
    
    arr = utils.ensure_np_array(arr)
    
    if npix is not None:
        arr = utils.pad_or_crop(arr, npix)
    
    if pxscl is not None:
        if isinstance(pxscl, u.Quantity):
            pxscl = pxscl.value
        vext = pxscl * arr.shape[0]/2
        hext = pxscl * arr.shape[1]/2
        extent = [-vext,vext,-hext,hext]
    else:
        extent=None
    
    norm = LogNorm(vmin=vmin,vmax=vmax) if lognorm else Normalize(vmin=vmin,vmax=vmax)
    
    im = ax.imshow(arr, cmap=cmap, norm=norm, extent=extent)
    if axlims is not None:
        ax.set_xlim(axlims1[:2])
        ax.set_ylim(axlims1[2:])
    ax.tick_params(axis='x', labelsize=9, rotation=30)
    ax.tick_params(axis='y', labelsize=9, rotation=30)
    ax.set_xlabel(xlabel)
    if patches: 
        for patch in patches:
            ax.add_patch(patch)
    ax.set_title(title)
    if grid: ax.grid()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    plt.close()
    
    if display_fig: 
        display(fig)
    if return_fig: 
        return fig,ax
    
def imshow2(arr1, arr2, 
            title1=None, title2=None,
            xlabel=None, xlabel1=None, xlabel2=None,
            npix=None, npix1=None, npix2=None,
            pxscl=None, pxscl1=None, pxscl2=None,
            axlims=None, axlims1=None, axlims2=None,
            grid=False, grid1=False, grid2=False,
            cmap1='magma', cmap2='magma',
            lognorm=False, lognorm1=False, lognorm2=False,
            vmin1=None, vmax1=None, vmin2=None, vmax2=None,
            patches1=None, patches2=None,
            display_fig=True, 
            return_fig=False, 
            figsize=(10,4), dpi=125, wspace=0.2):
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=dpi)
    
    if cp and isinstance(arr1, cp.ndarray): arr1 = arr1.get()
    if cp and isinstance(arr2, cp.ndarray): arr2 = arr2.get()
    
    npix1, npix2 = (npix, npix) if npix is not None else (npix1, npix2)
    if npix1 is not None: arr1 = pad_or_crop(arr1, npix1)
    if npix2 is not None: arr2 = pad_or_crop(arr2, npix2)
    
    pxscl1, pxscl2 = (pxscl, pxscl) if pxscl is not None else (pxscl1, pxscl2)
    if pxscl1 is not None:
        if isinstance(pxscl1, u.Quantity):
            vext = pxscl1.value * arr1.shape[0]/2
            hext = pxscl1.value * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
        else:
            vext = pxscl1 * arr1.shape[0]/2
            hext = pxscl1 * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
    else:
        extent1=None
        
    if pxscl2 is not None:
        if isinstance(pxscl2, u.Quantity):
            vext = pxscl2.value * arr2.shape[0]/2
            hext = pxscl2.value * arr2.shape[1]/2
            extent2 = [-pxscl2.value *arr2.shape[0]/2,vext,-hext,hext]
        else:
            vext = pxscl2 * arr2.shape[0]/2
            hext = pxscl2 * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
    else:
        extent2=None
    
    axlims1, axlims2 = (axlims, axlims) if axlims is not None else (axlims1, axlims2) # overide axlims
    xlabel1, xlabel2 = (xlabel, xlabel) if xlabel is not None else (xlabel1, xlabel2)
    
    norm1 = LogNorm(vmin=vmin1,vmax=vmax1) if lognorm1 or lognorm else Normalize(vmin=vmin1,vmax=vmax1)
    norm2 = LogNorm(vmin=vmin2,vmax=vmax2) if lognorm2 or lognorm else Normalize(vmin=vmin2,vmax=vmax2)
    
    # first plot
    im = ax[0].imshow(arr1, cmap=cmap1, norm=norm1, extent=extent1)
    if axlims1 is not None:
        ax[0].set_xlim(axlims1[:2])
        ax[0].set_ylim(axlims1[2:])
    if grid or grid1: ax[0].grid()
    ax[0].tick_params(axis='x', labelsize=9, rotation=30)
    ax[0].tick_params(axis='y', labelsize=9, rotation=30)
    ax[0].set_xlabel(xlabel1)
    if patches1: 
        for patch1 in patches1:
            ax[0].add_patch(patch1)
    ax[0].set_title(title1)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    
    # second plot
    im = ax[1].imshow(arr2, cmap=cmap2, norm=norm2, extent=extent2)
    if axlims2 is not None:
        ax[1].set_xlim(axlims2[:2])
        ax[1].set_ylim(axlims2[2:])
    if grid or grid2: ax[1].grid()
    ax[1].tick_params(axis='x', labelsize=9, rotation=30)
    ax[1].tick_params(axis='y', labelsize=9, rotation=30)
    ax[1].set_xlabel(xlabel2)
    if patches2: 
        for patch2 in patches2:
            ax[1].add_patch(patch2)
    ax[1].set_title(title2)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
        
    plt.subplots_adjust(wspace=wspace)
    
    plt.close()
    
    if display_fig: display(fig)
    if return_fig: return fig,ax

def imshow3(arr1, arr2, arr3,
            title1=None, title2=None, title3=None, titlesize=12,
            npix=None, npix1=None, npix2=None, npix3=None,
            pxscl=None, pxscl1=None, pxscl2=None, pxscl3=None, 
            axlims=None, axlims1=None, axlims2=None, axlims3=None,
            xlabel=None, xlabel1=None, xlabel2=None, xlabel3=None,
            cmap1='magma', cmap2='magma', cmap3='magma',
            lognorm=False, lognorm1=False, lognorm2=False, lognorm3=False,
            vmin1=None, vmax1=None, vmin2=None, vmax2=None, vmin3=None, vmax3=None, 
            patches1=None, patches2=None, patches3=None,
            grid=False, grid1=False, grid2=False, grid3=False,
            display_fig=True, 
            return_fig=False,
            figsize=(14,7), dpi=125, wspace=0.3):
    fig,ax = plt.subplots(nrows=1, ncols=3, figsize=figsize, dpi=dpi)
    
    if cp and isinstance(arr1, cp.ndarray): arr1 = arr1.get()
    if cp and isinstance(arr2, cp.ndarray): arr2 = arr2.get()
    if cp and isinstance(arr3, cp.ndarray): arr3 = arr3.get()
    
    npix1, npix2, npix3 = (npix, npix, npix) if npix is not None else (npix1, npix2, npix3)
    if npix1 is not None: arr1 = pad_or_crop(arr1, npix1)
    if npix2 is not None: arr2 = pad_or_crop(arr2, npix2)
    if npix3 is not None: arr2 = pad_or_crop(arr3, npix3)
    
    pxscl1, pxscl2, pxscl3 = (pxscl, pxscl, pxscl) if pxscl is not None else (pxscl1, pxscl2, pxscl3)
    if pxscl1 is not None:
        if isinstance(pxscl1, u.Quantity):
            vext = pxscl1.value * arr1.shape[0]/2
            hext = pxscl1.value * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
        else:
            vext = pxscl1 * arr1.shape[0]/2
            hext = pxscl1 * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
    else:
        extent1=None
        
    if pxscl2 is not None:
        if isinstance(pxscl2, u.Quantity):
            vext = pxscl2.value * arr2.shape[0]/2
            hext = pxscl2.value * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
        else:
            vext = pxscl2 * arr2.shape[0]/2
            hext = pxscl2 * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
    else:
        extent2=None
        
    if pxscl3 is not None:
        if isinstance(pxscl3, u.Quantity):
            vext = pxscl3.value * arr3.shape[0]/2
            hext = pxscl3.value * arr3.shape[1]/2
            extent3 = [-vext,vext,-hext,hext]
        else:
            vext = pxscl3 * arr3.shape[0]/2
            hext = pxscl3 * arr3.shape[1]/2
            extent3 = [-vext,vext,-hext,hext]
    else:
        extent3 = None
    
    axlims1, axlims2, axlims3 = (axlims, axlims, axlims) if axlims is not None else (axlims1, axlims2, axlims3) # overide axlims
    xlabel1, xlabel2, xlabel3 = (xlabel, xlabel, xlabel) if xlabel is not None else (xlabel1, xlabel2, xlabel3)
    
    norm1 = LogNorm(vmin=vmin1,vmax=vmax1) if lognorm1 or lognorm else Normalize(vmin=vmin1,vmax=vmax1)
    norm2 = LogNorm(vmin=vmin2,vmax=vmax2) if lognorm2 or lognorm else Normalize(vmin=vmin2,vmax=vmax2)
    norm3 = LogNorm(vmin=vmin3,vmax=vmax3) if lognorm3 or lognorm else Normalize(vmin=vmin3,vmax=vmax3)
    
    # first plot
    im = ax[0].imshow(arr1, cmap=cmap1, norm=norm1, extent=extent1)
    if axlims1 is not None:
        ax[0].set_xlim(axlims1[:2])
        ax[0].set_ylim(axlims1[2:])
    if grid or grid1: ax[0].grid()
    ax[0].tick_params(axis='x', labelsize=9, rotation=30)
    ax[0].tick_params(axis='y', labelsize=9, rotation=30)
    ax[0].set_xlabel(xlabel1)
    if patches1: 
        for patch1 in patches1:
            ax[0].add_patch(patch1)
    ax[0].set_title(title1)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    
    # second plot
    im = ax[1].imshow(arr2, cmap=cmap2, norm=norm2, extent=extent2)
    if axlims2 is not None:
        ax[1].set_xlim(axlims2[:2])
        ax[1].set_ylim(axlims2[2:])
    if grid or grid2: ax[1].grid()
    ax[1].tick_params(axis='x', labelsize=9, rotation=30)
    ax[1].tick_params(axis='y', labelsize=9, rotation=30)
    ax[1].set_xlabel(xlabel2)
    if patches2: 
        for patch2 in patches2:
            ax[1].add_patch(patch2)
    ax[1].set_title(title2)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    
    # second plot
    im = ax[2].imshow(arr3, cmap=cmap3, norm=norm3, extent=extent3)
    if axlims3 is not None:
        ax[2].set_xlim(axlims3[:2])
        ax[2].set_ylim(axlims3[2:])
    if grid or grid3: ax[1].grid()
    ax[2].tick_params(axis='x', labelsize=9, rotation=30)
    ax[2].tick_params(axis='y', labelsize=9, rotation=30)
    ax[2].set_xlabel(xlabel3)
    if patches3: 
        for patch3 in patches3:
            ax[2].add_patch(patch3)
    ax[2].set_title(title3)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
        
    plt.subplots_adjust(wspace=wspace)
    
    plt.close()
    
    if display_fig: display(fig)
    if return_fig: return fig,ax
    