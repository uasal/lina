from .math_module import xp
from . import utils, scc
from . import imshows

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

import poppy

def create_zernike_modes(pupil_mask, nmodes=15):
    
    zernikes = poppy.zernike.arbitrary_basis(pupil_mask, nterms=nmodes, outside=0)

    return zernikes

def calibrate(sysi, 
              calibration_modes, calibration_amp,
              control_mask, 
              scc_fun=None, scc_params=None,
              plot=False,
              ):
    """
    This function will compute the Jacobian for EFC using either the system model 
    or the SCC estimation function. If SCC is used, this function can be used with a real instrument. 

    Parameters
    ----------
    sysi : object
        The object of a system interface with methods for DM control and image capture
    calibration_modes : xp.ndarray
        2D array of modes to be calibrated in the Jacobian, shape of (Nmodes, Nact**2)
    calibration_amp : float
        amplitude to be applied to each calibration mode while Jacobian is computed
    control_mask : xp.ndarray
        the boolean mask defining the focal plane pixels in the control region
    plot : bool, optional
        whether of not to plot the RMS response of DM actuators, by default False
    """

    start = time.time()
    
    amps = np.linspace(-calibration_amp, calibration_amp, 2) # for generating a negative and positive actuator poke
    
    Nmodes = calibration_modes.shape[0]
    Nmask = int(control_mask.sum())
    
    responses = xp.zeros((2*Nmask, Nmodes))
    print('Calculating Jacobian: ')
    for i,mode in enumerate(calibration_modes):
        response = 0
        for amp in amps:
            mode = mode.reshape(sysi.Nact,sysi.Nact)

            if scc_fun is None: # using the model to build the Jacobian
                sysi.add_dm(amp*mode)
                wavefront = sysi.calc_psf()
                response += amp * wavefront.flatten() / (2*np.var(amps))
                sysi.add_dm(-amp*mode)
            elif scc_fun is not None and scc_params is not None:
                sysi.add_dm(amp*mode)
                wavefront = scc_fun(sysi, **scc_params)
                response += amp * wavefront.flatten() / (2*np.var(amps))
                sysi.add_dm(-amp*mode)

        responses[::2,i] = response[control_mask.ravel()].real
        responses[1::2,i] = response[control_mask.ravel()].imag

        print('\tCalculated response for mode {:d}/{:d}. Elapsed time={:.3f} sec.'.format(i+1, Nmodes, time.time()-start), end='')
        print("\r", end="")
    
    print()
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    if plot:
        total_response = responses[::2] + 1j*responses[1::2]
        dm_response = total_response.dot(xp.array(calibration_modes))
        dm_response = xp.sqrt(xp.mean(xp.abs(dm_response)**2, axis=0)).reshape(sysi.Nact, sysi.Nact)
        imshows.imshow1(dm_response, lognorm=True, vmin=dm_response.max()*1e-2)

    return responses







