from .math_module import xp, _scipy
from . import utils
from . import imshows
import time
import copy
import matplotlib.pyplot as plt

def estimate_coherent(mod_image, unmod_image, scc_ref_image, r_npix, shift, 
                      dark_mask=None, plot=False):
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
    
    if dark_mask is not None:
        im *= dark_mask
    
    mod_fft = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(mod_image), norm='ortho'))
    unmod_fft = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(unmod_image), norm='ortho'))

    fft_diff = mod_fft - unmod_fft

    if plot:
        plt.figure(figsize=(18, 5))
        plt.subplot(131)
        plt.title("Modulated Image FFT")
        plt.imshow(utils.ensure_np_array(xp.log10(xp.abs(mod_fft) ** 2)))
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(132)
        plt.title("Unmodulated Image FFT")
        plt.imshow(utils.ensure_np_array(xp.log10(xp.abs(unmod_fft) ** 2)))
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(133)
        plt.title("FFT Difference")
        plt.imshow(utils.ensure_np_array(xp.log10(xp.abs(fft_diff) ** 2)))
        plt.colorbar(fraction=0.046, pad=0.04)

    fft_shifted = _scipy.ndimage.shift(fft_diff, shift)
    
    x = xp.linspace(-mod_image.shape[0]//2, mod_image.shape[0]//2-1, mod_image.shape[0]) + 1/2
    x,y = xp.meshgrid(x,x)
    
    r = xp.sqrt(x ** 2 + y ** 2)
    mask = r < r_npix
    fft_masked = mask * fft_shifted
    
    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.title("Shifted FFT Difference")
        plt.imshow(utils.ensure_np_array(xp.log10(xp.abs(fft_shifted) ** 2)))
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(122)
        plt.title("Cropped+Shifted FFT Difference")
        plt.imshow(utils.ensure_np_array(xp.log10(xp.abs(fft_masked) ** 2)))
        plt.colorbar(fraction=0.046, pad=0.04)
    
    E_est = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(fft_masked), norm='ortho'))

    if dark_mask is not None:
        E_est *= dark_mask
        
    E_est /= xp.sqrt(scc_ref_image)

    return E_est