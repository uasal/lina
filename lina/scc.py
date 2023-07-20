from .math_module import xp, _scipy
from . import utils
from . import imshows

from IPython.display import display, clear_output

def estimate_coherent(sysi, r_npix=0, shift=(0,0), plot=False):
    '''
    r_npix:
        radius of sidebands in units of pixels
    shift:
        location of sideband centers in pixels (from center of array)
    '''
    
    im = sysi.snap()
    
    im_fft = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(im)))
    if plot:
        imshows.imshow2(xp.abs(im_fft), xp.angle(im_fft), lognorm1=True)
    im_fft_shift = _scipy.ndimage.shift(im_fft, shift)
    
    x = xp.linspace(-im.shape[0]//2, im.shape[0]//2-1, im.shape[0]) + 1/2
    x,y = xp.meshgrid(x,x)
    
    r = xp.sqrt(x**2 + y**2)
    mask = r<r_npix
    im_fft_masked = mask*xp.abs(im_fft_shift)
    
    if plot:
        imshows.imshow3(mask, xp.abs(im_fft_shift), im_fft_masked, lognorm2=True, lognorm3=True)
    
    E_est = xp.fft.fftshift(xp.fft.ifft2(im_fft_masked))
    
    return E_est

def estimate_incoherent():
    '''
    FIXME
    '''
    
    return E_est
