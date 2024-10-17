from .math_module import xp, ensure_np_array
from lina.imshows import imshow1, imshow2, imshow3
import lina.utils as utils

import numpy as np
import astropy.units as u
import copy
from IPython.display import display, clear_output

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

    response_matrix = responses.T

    return response_matrix

def calibrate_with_fsm(I, control_mask, dm_modes=None, amps=5e-9, plot=False):
    # time.sleep(2)
    Nmask = int(control_mask.sum())
    Nmodes = dm_modes.shape[0] + 2 if dm_modes is not None else 2
    if np.isscalar(amps):
        amps = [amps] * Nmodes

    responses = xp.zeros((Nmodes, Nmask))
    for i in range(Nmodes):
        amp = amps[i]

        if i==0: # assumes this is the Tip mode so applies it to the FSM
            mode = I.TTM[0]
            amp_pv = amp / I.tt_pv_to_rms
            amp_as = (np.arctan(amp_pv / I.fsm_beam_diam.to_value(u.m))*u.radian).to(u.arcsec) 
            I.add_fsm(tip=amp_as, tilt=0*u.arcsec)
            im_pos = I.snap_locam()
            I.add_fsm(tip=-2*amp_as, tilt=0*u.arcsec)
            im_neg = I.snap_locam()
            I.add_fsm(tip=amp_as, tilt=0*u.arcsec)
        elif i==1: # assumes this is the Tilt mode so applies it to the FSM
            mode = I.TTM[1]
            amp_pv = amp / I.tt_pv_to_rms
            amp_as = (np.arctan(amp_pv / I.fsm_beam_diam.to_value(u.m))*u.radian).to(u.arcsec) 
            I.add_fsm(tip=0*u.arcsec, tilt=amp_as)
            im_pos = I.snap_locam()
            I.add_fsm(tip=0*u.arcsec, tilt=-2*amp_as)
            im_neg = I.snap_locam()
            I.add_fsm(tip=0*u.arcsec, tilt=amp_as)
        else: # use DM for all higher order modes
            mode = dm_modes[i-2]
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

def update_locam_delta(response_matrix, modal_matrix, control_mask, dh_channel, locam_delta_channel,):
    del_ref_im = np.zeros(locam_delta_channel.shape)
    del_ref_im[control_mask] = response_matrix.dot(modal_matrix.dot(1e-6*dh_channel.grab_latest().ravel())/1024)
    locam_delta_channel.write(del_ref_im)
    return

def single_iteration(I,
                     locam_ref_channel,
                     locam_delta_channel,  
                     control_matrix, 
                     modal_matrix,
                     control_mask, 
                     gain=1/2,
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
    modal_coeff *= gain
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

def run_model(M, 
              static_wfe, 
              ref_im, 
              control_mask, 
              control_matrix, 
              time_series, 
              wfe_modes, 
              dm_modes,
              gain=1/2,  
              plot=False, 
              plot_all=False,
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
    dm_modes : _type_
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
    diff_ims = xp.zeros((Nitr, M.nlocam, M.nlocam))
    fsm_commands = xp.zeros((Nitr, 2))
    coro_ims = xp.zeros((Nitr, M.npsf, M.npsf))
    dm_commands = xp.zeros((Nitr, M.Nact, M.Nact))
    injected_wfes = xp.zeros((Nitr, time_series[1:, 0].shape[0]))
    
    for i in range(Nitr):
        if i==0:
            tt_coeff_rms = xp.array([0,0])
            del_dm_command = xp.zeros_like(M.DM.command)
        else:
            # compute the DM command with the image based on the time delayed wavefront
            modal_coeff = - gain * control_matrix.dot(del_im[control_mask])
            tt_coeff_rms = modal_coeff[:2]
            del_dm_command = xp.sum( modal_coeff[2:, None, None] * dm_modes, axis=0)
        # print(gain)
        tt_coeff_pv = tt_coeff_rms / M.tt_pv_to_rms
        tt_coeff_as = (np.arctan( ensure_np_array(tt_coeff_pv) / M.fsm_beam_diam.to_value(u.m))*u.radian).to(u.arcsec) / 2
        M.add_fsm(tip=tt_coeff_as[0], tilt=tt_coeff_as[1])
        M.add_dm(del_dm_command)

        # apply the new wavefront to simulate a time delay 
        lo_wfe = xp.sum( time_series[1:, i, None, None] * wfe_modes, axis=0)
        M.setattr('WFE', static_wfe * xp.exp(1j * 2*np.pi/M.wavelength_c.to_value(u.m) * lo_wfe) )
        locam_im = M.snap_locam()
        coro_im = M.snap()
        del_im = locam_im - ref_im

        llowfsc_ims[i] = copy.copy(locam_im)
        diff_ims[i] = copy.copy(del_im)
        coro_ims[i] = copy.copy(coro_im)
        fsm_commands[i] = copy.copy(M.fsm_rms)
        dm_commands[i] = copy.copy(M.get_dm())
        injected_wfes[i] = copy.copy(time_series[1:, i])

        if plot or plot_all:
            imshow3(locam_im, del_im, coro_im, 
                    'LLOWFSC Image', 'Difference Image',
                    cmap1='magma', cmap2='magma',
                    lognorm3=True, vmin3=1e-9, )
            rms_wfe = xp.sqrt(xp.mean(xp.square( lo_wfe[M.APMASK] )))
            vmax_pup = 2*rms_wfe
            pupil_cmap = 'viridis'
            imshow3(lo_wfe, M.FSM, M.get_dm(), 
                    f'Current WFE: {rms_wfe:.2e}\nTime = {time_series[0][i]:.3f}s', 
                    'LLOWFSC FSM Command', 'LLOWFSC DM Command',
                    vmin1=-vmax_pup, vmax1=vmax_pup, 
                    cmap1=pupil_cmap, cmap2=pupil_cmap, cmap3=pupil_cmap,
                    )
            
            if not plot_all: clear_output(wait=True)

    sim_dict = {
        'llowfsc_ims':llowfsc_ims,
        'diff_ims':diff_ims, 
        'injected_wfes':injected_wfes,
        'llowfsc_ref':ref_im,
        'wfe_modes':wfe_modes, 
        'coro_ims':coro_ims,
        'fsm_commands':fsm_commands,
        'dm_commands':dm_commands,
    }
    
    return sim_dict







