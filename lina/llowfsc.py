from .math_module import xp, _scipy, ensure_np_array
import lina.utils as utils
from lina.imshows import imshow1, imshow2, imshow3

import numpy as np
import astropy.units as u
import copy
from IPython.display import display, clear_output
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

def record_chop_ims(I, mode, amp, Nchops, plot=False):
    chops = xp.zeros((3*Nchops, I.nlocam, I.nlocam))
    for i in range(Nchops):
        # First image at the 0 point
        zero_im = I.snap_locam()

        # Second image at the positive chop
        I.add_dm(amp*mode)
        pos_im = I.snap_locam()
        I.add_dm(-amp*mode)

        # Fourth image at negative chop
        I.add_dm(-amp*mode)
        neg_im = I.snap_locam()
        I.add_dm(amp*mode)

        if plot:
            imshow3(zero_im, pos_im, neg_im)

        chops[3*i] = zero_im
        chops[3*i+1] = pos_im
        chops[3*i+2] = neg_im

    return chops

def make_shear_chops(ref_locam_im, shear_pix=1, plot=False):
    nlocam = ref_locam_im.shape[0]
    shear_chops = xp.zeros((2, nlocam, nlocam))
    shear_chops[0] = ( _scipy.ndimage.shift(ref_locam_im, (0,shear_pix), order=5) - ref_locam_im ) / shear_pix
    shear_chops[1] = ( _scipy.ndimage.shift(ref_locam_im, (shear_pix,0), order=5) - ref_locam_im ) / shear_pix
    if plot: imshow2(shear_chops[0], shear_chops[1])
    return shear_chops

def make_response_matrix(zernike_chops, control_mask, flux_mode=None, shear_chops=None,):
    """_summary_

    Parameters
    ----------
    zernike_chop_ims : np.ndarray
        Data cube containing the difference images for each zernike mode
    control_mask : _type_
        _description_
    flux_mode : _type_, optional
        _description_, by default None
    shear_chops : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """

    Nmask = int(control_mask.sum())
    Nz_modes = zernike_chops.shape[0]
    Nmodes = Nz_modes + 2
    if shear_chops is None:
        shear_chop = xp.zeros_like(zernike_chops[0])
        shear_chops = xp.array([shear_chop, shear_chop])
    # if flux_mode is None:
    #     flux_mode = np.zeros_like(zernike_chops[0])
    
    response_matrix = xp.zeros((Nmask, Nmodes))
    for i in range(Nz_modes):
        response_matrix[:,i] = zernike_chops[i][control_mask]
    response_matrix[:,-2] = shear_chops[0][control_mask]
    response_matrix[:,-1] = shear_chops[1][control_mask]
    # response_matrix[:,-1] = flux_mode[control_mask]

    return response_matrix

def reconstruct(locam_im, control_matrix, ref_im, verbose=False):

    del_im = locam_im - ref_im
    coeff = control_matrix.dot(del_im)

    return coeff

def update_locam_delta(response_matrix, modal_matrix, control_mask, dh_channel, locam_delta_channel,):
    del_ref_im = np.zeros(locam_delta_channel.shape)
    del_ref_im[control_mask] = response_matrix.dot(modal_matrix.dot(1e-6*dh_channel.grab_latest().ravel())/1024)
    locam_delta_channel.write(del_ref_im)
    return

def inject_wfe(wfe_time_series, wfe_modes, freq, wfe_channel):
    Nsamps = wfe_time_series.shape[1]
    try:
        print('Injecting WFE ...')
        i = 0
        while i<Nsamps+1:
            if i==Nsamps:
                i = 0
            wfe = np.sum( wfe_time_series[:, i, None, None] * wfe_modes, axis=0)
            wfe_channel.write(1e6 * wfe)
            time.sleep(1/freq)
            i += 1
            # print(i)
    except KeyboardInterrupt:
        print('Stopped injecting WFE.')
        wfe_channel.write(np.zeros(wfe_channel.shape))

def calibrate_without_fsm(I, control_mask, dm_modes, amps=5e-9, plot=False):
    # time.sleep(2)
    Nmask = int(control_mask.sum())
    Nmodes = dm_modes.shape[0]
    if np.isscalar(amps):
        amps = [amps] * Nmodes

    if isinstance(control_mask, np.ndarray):
        responses = np.zeros((Nmodes, Nmask))
    else:
        responses = xp.zeros((Nmodes, Nmask))
    
    start = time.time()
    for i in range(Nmodes):
        amp = amps[i]
        mode = dm_modes[i]

        I.add_dm(amp*mode)
        im_pos = I.snap_locam()
        I.add_dm(-2*amp*mode)
        im_neg = I.snap_locam()
        I.add_dm(amp*mode)

        diff = im_pos - im_neg
        responses[i] = copy.copy(diff)[control_mask]/(2 * amp)
        
        if plot:
            imshow3(amp*mode, im_pos, diff, f'Mode {i+1}', 'Absolute Image', 'Difference', cmap1='viridis')
        
        print(f"\tCalibrated mode {i+1:d}/{dm_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")

    response_matrix = responses.T

    return response_matrix

def single_iteration(
    I,
    locam_ref_channel,
    locam_delta_channel,  
    gain_channel, 
    control_matrix, 
    modal_matrix,
    control_mask, 
    thresh=0,
    leakage=0.0, 
    plot=False,
    clear=False,
    ):

    image = I.snap_locam()
    del_im = image - (locam_ref_channel.grab_latest() + locam_delta_channel.grab_latest())

    # compute the DM command with the image based on the time delayed wavefront
    modal_coeff = -control_matrix.dot(del_im[control_mask])
    modal_coeff *= np.abs(modal_coeff) >= thresh
    modal_coeff *= gain_channel.grab_latest()[0]
    del_dm_command = modal_matrix.T.dot(modal_coeff).reshape(I.Nact,I.Nact)
    # I.add_dm(del_dm_command)

    total_command = (1-leakage) * ensure_np_array(I.get_dm()) + del_dm_command
    I.set_dm(total_command)

    if plot:
        dm_command = I.get_dm()
        pv_stroke = xp.max(dm_command) - xp.min(dm_command)
        rms_stroke = xp.sqrt(xp.mean(xp.square(dm_command[I.dm_mask])))
        imshow3(del_im, del_dm_command, dm_command, 
                'Measured Difference Image', 
                'Computed DM Correction',
                f'PV Stroke = {1e9*pv_stroke:.1f}nm\nRMS Stroke = {1e9*rms_stroke:.1f}nm', 
                cmap1='magma', cmap2='viridis', cmap3='viridis',
                )
        if clear: clear_output(wait=True)





