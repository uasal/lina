from .math_module import xp, _scipy
from . import utils
from . import imshows
import time

from IPython.display import display, clear_output

def estimate_coherent(sysi, r_npix=0, shift=(0,0), dark_mask=None, plot=False):
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
        imshows.imshow3(mask, xp.abs(im_fft_shift), xp.abs(im_fft_masked), lognorm2=True, lognorm3=True)
    
    E_est = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(im_fft_masked)))

    if dark_mask is not None:
        E_est *= dark_mask

    norm = (xp.abs(E_est) ** 2).max()
    E_est *= xp.sqrt(im_max / norm)

    return E_est

def estimate_incoherent():
    '''
    FIXME
    '''
    
    return E_est

def build_jacobian(sysi, 
                   epsilon, 
                   control_mask,
                   control_modes,
                   plot=False,
                   **scc_kwargs,
                   ):
    '''
    This can be done on the actual testbed with individual pokes
    '''
    start = time.time()
    
    amps = xp.linspace(-epsilon, epsilon, 2) # for generating a negative and positive actuator poke
    
    dm_mask = sysi.dm_mask.flatten()
#     if hasattr(sysi, 'bad_acts'):
#         dm_mask[sysi.bad_acts] = False
    
    Nacts = int(dm_mask.sum())
    Nmask = int(control_mask.sum())
    
    num_modes = control_modes.shape[0]
    modes = control_modes 
    
    responses = xp.zeros((2*Nmask, num_modes))
    count = 0

    print('Calculating Jacobian: ')
    for i in range(num_modes):
        if dm_mask[i]:
            response = 0
            for amp in amps:
                mode = modes[i].reshape(sysi.Nact,sysi.Nact)

                sysi.add_dm(amp*mode)
                wavefront = estimate_coherent(sysi, **scc_kwargs)
                wavefront *= control_mask
                response += amp * wavefront.flatten() / (2*xp.var(amps))
                sysi.add_dm(-amp*mode)
            
            responses[::2,count] = response[control_mask.ravel()].real
            responses[1::2,count] = response[control_mask.ravel()].imag
            
            print('\tCalculated response for mode {:d}/{:d}. Elapsed time={:.3f} sec.'.format(count+1, Nacts, time.time()-start), end='')
            print("\r", end="")
            count += 1
        else:
            pass
    print()
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    return responses
