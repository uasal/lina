from .math_module import xp, _scipy
from . import utils
from . import imshows
import time
import copy

import numpy as np

from IPython.display import display, clear_output

def estimate_coherent(sysi, r_npix=0, shift=(0,0), dark_mask=None, plot=False, plot_est=False):
    '''
    r_npix:
        radius of sidebands in units of pixels
    shift:
        location of sideband centers in pixels (from center of array)
    '''

    im = sysi.snap()
    
    if dark_mask is not None:
        im *= dark_mask

    im_max = im.max()
    
    im_fft = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(im), norm='ortho'))
    # im_fft_sum = xp.sum(xp.abs(im_fft))
    
    if plot:
        imshows.imshow2(xp.abs(im_fft), xp.angle(im_fft), lognorm1=True)
    im_fft_shift = _scipy.ndimage.shift(im_fft, shift)
    
    x = xp.linspace(-im.shape[0]//2, im.shape[0]//2-1, im.shape[0]) + 1/2
    x,y = xp.meshgrid(x,x)
    
    r = xp.sqrt(x**2 + y**2)
    mask = r<r_npix
    im_fft_masked = mask*im_fft_shift
    
    # im_fft_masked_sum = xp.sum(xp.abs(im_fft_masked))
    # im_fft_masked *= xp.sqrt((im_fft_sum-im_fft_masked_sum)/im_fft_masked_sum)
    
    if plot:
        fig,ax = imshows.imshow3(mask, xp.abs(im_fft_shift), xp.abs(im_fft_masked), lognorm2=True, lognorm3=True,
                                 display_fig=False, return_fig=True)
        ax[1].grid()
        ax[1].set_xticks(np.linspace(0, im_fft_shift.shape[0], 7))
        ax[1].set_yticks(np.linspace(0, im_fft_shift.shape[0], 7))

        display(fig)
    
    E_est = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(im_fft_masked), norm='ortho'))

    if dark_mask is not None:
        E_est *= dark_mask

    norm = (xp.abs(E_est) ** 2).max()
    E_est *= xp.sqrt(im_max / norm)

    if plot or plot_est:
        imshows.imshow2(xp.abs(E_est)**2, xp.angle(E_est), lognorm1=True, pxscl=sysi.psf_pixelscale_lamD)

    return E_est

def estimate_incoherent():
    '''
    FIXME
    '''
    
    return E_est


def estimate_coherent_mod(sysi, 
                        #   mod_image, unmod_image, scc_ref_image, 
                          r_npix, shift, 
                          dark_mask=None, 
                          plot=False,):
    '''
    mod_image:
        SCC modulated science image taken using an SCC stop 
    unmod_image:
        Unmodulated science image taken using a standard Lyot stop
    scc_ref_image:
        Reference image taken using just the SCC's pinhole for normalization of the estimated electric field
    r_npix:
        Radius of sidebands in units of pixels
    shift:
        Location of sideband centers in pixels (from center of array)
    dark_mask:
        Dark hole mask (optional)
    '''

    sysi.use_scc()
    mod_image = sysi.snap()

    sysi.use_scc(False)
    unmod_image = sysi.snap()

    sysi.block_lyot()
    scc_ref_image = sysi.snap()
    sysi.block_lyot(False)
    
    if dark_mask is not None:
        mod_image *= dark_mask
        unmod_image *= dark_mask
        mask_fft = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(dark_mask), norm='ortho'))

    mod_fft = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(mod_image), norm='ortho'))
    unmod_fft = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(unmod_image), norm='ortho'))

    fft_diff = mod_fft - unmod_fft
    # if dark_mask is not None:
    #     fft_diff -= mask_fft

    if plot:
        imshows.imshow3(xp.abs(mod_fft), xp.abs(unmod_fft), xp.abs(fft_diff), lognorm=True, )

    fft_shifted = _scipy.ndimage.shift(fft_diff, shift)
    
    x = xp.linspace(-mod_image.shape[0]//2, mod_image.shape[0]//2-1, mod_image.shape[0]) + 1/2
    x,y = xp.meshgrid(x,x)
    
    r = xp.sqrt(x ** 2 + y ** 2)
    mask = r < r_npix
    fft_masked = mask * fft_shifted
    
    if plot:
        imshows.imshow2(xp.abs(fft_shifted) ** 2, xp.abs(fft_masked) ** 2, lognorm=True)
    
    E_est = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(fft_masked), norm='ortho'))

    if dark_mask is not None:
        E_est *= dark_mask
        
    E_est /= xp.sqrt(scc_ref_image)

    return E_est


