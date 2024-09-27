from .math_module import xp
from lina.imshows import imshow1, imshow2, imshow3
import lina.utils as utils

import numpy as np
import astropy.units as u
import copy
from IPython.display import display, clear_output

def calibrate(I, calib_modes, control_mask, amps=5e-9, plot=False):
    # time.sleep(2)
    Nmodes = calib_modes.shape[0]
    Nmask = int(control_mask.sum())
    if np.isscalar(amps):
        amps = [amps] * Nmodes

    responses = xp.zeros((Nmodes, Nmask))
    for i in range(Nmodes):
        mode = calib_modes[i]
        amp = amps[i]

        I.add_dm(amp*mode)
        im_pos = I.snap_locam()
        I.add_dm(-2*amp*mode)
        im_neg = I.snap_locam()
        I.add_dm(amp*mode)

        diff = im_pos - im_neg
        responses[i] = copy.copy(diff)[control_mask]/(2 * amp)

        if plot:
            imshow3(amp*mode, im_pos, diff, f'Mode {i+1}', 'Absolute Image', 'Difference', cmap1='viridis')

    response_matrix = responses.T

    return response_matrix

def single_iteration(I,
                    ref_im, 
                    control_matrix, 
                    control_modes,
                    control_mask, 
                    gain=1/2,
                    thresh=0,
                    plot=False,
                    clear=False,
                    ):

    image = I.snap_locam()
    del_im = image - ref_im

    # compute the DM command with the image based on the time delayed wavefront
    modal_coeff = control_matrix.dot(del_im[control_mask])
    modal_coeff *= np.abs(modal_coeff) >= thresh
    modal_coeff *= gain
    del_dm_command = -control_modes.dot(modal_coeff).reshape(I.Nact,I.Nact)
    # if reverse_dm_parity: del_dm_correction = xp.rot90(xp.rot90(del_dm_correction))
    I.add_dm(del_dm_command)

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

def run_model(M, 
              static_wfe, 
              ref_im, 
              control_mask, 
              control_matrix, 
              control_modes, 
              time_series, 
              wfe_modes, 
              gain=1/2,  
              reverse_dm_parity=False,
              plot=False, 
              plot_all=False,
              return_coro_ims=False,
              ):
    """_summary_

    Parameters
    ----------
    sysi : _type_
        _description_
    ref_im : _type_
        _description_
    control_matrix : _type_
        _description_
    control_modes : _type_
        _description_
    time_series_coeff : _type_
        _description_
    zernike_modes : _type_
        _description_
    plot : bool, optional
        _description_, by default False
    """
    print(f'Starting LLOWFSC control-loop simulation')

    Nitr = time_series.shape[1]
    llowfsc_ims = xp.zeros((Nitr, M.nlocam, M.nlocam))
    llowfsc_commands = xp.zeros((Nitr, M.Nact, M.Nact))
    if return_coro_ims: coro_ims = xp.zeros((Nitr, M.npsf, M.npsf))

    # prior to the first iteration, compute the initial image the first DM commands will be computed from
    lo_wfe = xp.sum( time_series[1:, 0, None, None] * wfe_modes, axis=0)
    M.setattr('WFE', static_wfe * xp.exp(1j * 2*np.pi/M.wavelength_c.to_value(u.m) * lo_wfe) ) 

    locam_im = M.snap_locam()
    del_im = locam_im - ref_im
    llowfsc_ims[0] = copy.copy(locam_im)
    total_command = 0.0
    total_coeff = 0.0
    for i in range(Nitr-1):
        # apply the new wavefront for the current iteration
        lo_wfe = xp.sum( time_series[1:, i+1, None, None] * wfe_modes, axis=0)
        M.setattr('WFE', static_wfe * xp.exp(1j * 2*np.pi/M.wavelength_c.to_value(u.m) * lo_wfe) )

        # compute the DM command with the image based on the time delayed wavefront
        modal_coeff = - gain * control_matrix.dot(del_im[control_mask]) 
        total_coeff += modal_coeff
        # del_dm_command = -control_modes.dot(modal_coeff).reshape(M.Nact,M.Nact)
        del_dm_command = xp.sum( modal_coeff[:, None, None] * control_modes, axis=0)
        if reverse_dm_parity:
            del_dm_command = xp.rot90(xp.rot90(del_dm_command))
        total_command += del_dm_command / 2 # divide by 2 for reflection
        M.add_dm(del_dm_command / 2)

        # compute the new LLOWFSC image to be used on the next iteration
        locam_im = M.snap_locam()
        del_im = locam_im - ref_im
        llowfsc_ims[i] = copy.copy(locam_im)
        llowfsc_commands[i] = copy.copy(total_command)
        if return_coro_ims: 
            coro_im = M.snap()
            coro_ims[i] = copy.copy(coro_im)

        if plot or plot_all:
            if return_coro_ims:
                imshow3(locam_im, del_im, coro_im, 
                        'LLOWFSC Image', 'Difference Image',
                        cmap1='magma', cmap2='magma',
                        lognorm3=True, vmin3=1e-9, 
                        )
            else: 
                imshow2(locam_im, del_im,
                        'LLOWFSC Image', 'Difference Image',
                        cmap1='magma', cmap2='magma',
                        )
            rms_wfe = xp.sqrt(xp.mean(xp.square( lo_wfe[M.APMASK] )))
            pv_stroke = xp.max(total_command) - xp.min(total_command)
            rms_stroke = xp.sqrt(xp.mean(xp.square( total_command[M.dm_mask] )))
            vmax_pup = 2*rms_wfe
            pupil_cmap = 'viridis'
            imshow3(lo_wfe, del_dm_command, total_command,
                    f'Current WFE: {rms_wfe:.2e}\nTime = {time_series[0][i+1]:.3f}s', 
                    'Computed DM Correction',
                    f'Total DM Command\nPV = {1e9*pv_stroke:.1f}nm, RMS = {1e9*rms_stroke:.1f}nm', 
                    vmin1=-vmax_pup, vmax1=vmax_pup, 
                    cmap1=pupil_cmap, cmap2=pupil_cmap, cmap3=pupil_cmap,
                    )
            
            if not plot_all: clear_output(wait=True) 
    if return_coro_ims:
        return llowfsc_ims, llowfsc_commands, coro_ims
    else:
        return llowfsc_ims, llowfsc_commands







