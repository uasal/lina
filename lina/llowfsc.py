from .math_module import xp, xcipy, ensure_np_array
from lina import utils, shmim_utils

import numpy as np
import astropy.units as u
import copy
from IPython.display import display, clear_output
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

def calibrate_without_fsm(
        CAMLO_STREAM,
        DM_STREAM,
        NFRAMES, 
        dm_modes, 
        control_mask, 
        amps=2.5e-9, 
        delay=0.05,
        plot=False,
    ):
    
    Nmask = int(control_mask.sum())
    Nmodes = dm_modes.shape[0]
    Nact = dm_modes.shape[1]
    Ncamlo = CAMLO_STREAM.shape[0]
    if np.isscalar(amps):
        amps = [amps] * Nmodes

    response_matrix = np.zeros((Nmask, Nmodes))
    response_cube = np.zeros((Nmodes, Ncamlo, Ncamlo))
    
    start = time.time()
    for i in range(Nmodes):
        amp = amps[i]
        mode = amp*dm_modes[i]

        DM_STREAM.write( mode * 1e6 )
        time.sleep(delay)
        im_pos = np.mean( CAMLO_STREAM.grab_many(NFRAMES), axis=0)

        DM_STREAM.write( -mode * 1e6 )
        time.sleep(delay)
        im_neg = np.mean( CAMLO_STREAM.grab_many(NFRAMES), axis=0)

        DM_STREAM.write(np.zeros((Nact,Nact)))

        diff = im_pos - im_neg
        response_cube[i] = copy.copy(diff) / (2 * amp)
        response_matrix[:,i] = copy.copy(diff)[control_mask] / (2 * amp)
        
        if plot: 
            print(f"Calibrated mode {i+1:d}/{Nmodes:d} in {time.time()-start:.3f}s", end='')
            utils.imshow(
                [amp*mode, im_pos, diff], 
                titles=[f'Mode {i+1}', 'Positive Chop Image', 'Difference'], 
                cmaps=['viridis'],
            )
        else:
            print(f"\tCalibrated mode {i+1:d}/{Nmodes:d} in {time.time()-start:.3f}s", end='')
            print("\r", end="")

    return response_matrix, response_cube

def calibrate_with_fsm(
        CAMLO_STREAM,
        FSM_STREAM,
        DM_STREAM,
        NFRAMES, 
        dm_zer_modes, 
        control_mask, 
        fsm_beam_diam,
        amps=2.5e-9,
        include_factor_2=False,
        flux_norm_coeff=None, 
        delay=0.05,
        plot=False,
    ):
    
    Nmask = int(control_mask.sum())
    Nmodes = 2 + dm_zer_modes.shape[0]
    Nact = DM_STREAM.shape[0]
    Ncamlo = CAMLO_STREAM.shape[0]
    if np.isscalar(amps): amps = [amps] * Nmodes

    response_matrix = np.zeros((Nmask, Nmodes))
    response_cube = np.zeros((Nmodes, Ncamlo, Ncamlo))
    
    start = time.time()
    for i in range(Nmodes):
        print(f"\tCalibrating Zernike mode {i+1:d}/{Nmodes:d}s", end='')
        print("\r", end="")
        if i==0:
            amp = amps[i]
            amp_as = utils.tt_rms_to_as(amp, fsm_beam_diam)
            fsm_command = np.array([0, amp_as, 0])
            shmim_utils.write(FSM_STREAM, fsm_command)
            time.sleep(delay)
            im_pos = shmim_utils.stack(CAMLO_STREAM, NFRAMES)
            shmim_utils.write(FSM_STREAM, -fsm_command)
            time.sleep(delay)
            im_neg = shmim_utils.stack(CAMLO_STREAM, NFRAMES)
            shmim_utils.write(FSM_STREAM, [0,0,0])
        elif i==1:
            amp = amps[i]
            amp_as = utils.tt_rms_to_as(amp, fsm_beam_diam)
            fsm_command = np.array([0, 0, amp_as])
            shmim_utils.write(FSM_STREAM, fsm_command)
            time.sleep(delay)
            im_pos = shmim_utils.stack(CAMLO_STREAM, NFRAMES)
            shmim_utils.write(FSM_STREAM, -fsm_command)
            time.sleep(delay)
            im_neg = shmim_utils.stack(CAMLO_STREAM, NFRAMES)
            shmim_utils.write(FSM_STREAM, [0,0,0])
        else:
            amp = amps[i]
            mode = amp*dm_zer_modes[i-2]
            DM_STREAM.write( mode * 1e6 )
            time.sleep(delay)
            im_pos = shmim_utils.stack(CAMLO_STREAM, NFRAMES)
            DM_STREAM.write( -mode * 1e6 )
            time.sleep(delay)
            im_neg = shmim_utils.stack(CAMLO_STREAM, NFRAMES)
            DM_STREAM.write(np.zeros((Nact,Nact)))

        diff = im_pos - im_neg
        response_cube[i] = copy.copy(diff) / (2 * amp)
        response_matrix[:,i] = copy.copy(diff)[control_mask] / (2 * amp)

        if plot: 
            utils.imshow(
                [im_pos, im_neg, diff], 
                titles=['Positive Chop', 'Negative Chop', 'Difference'], 
            )
    # response_matrix = response_matrix.T

    if include_factor_2: # in case you want to account for reflection off your FSM/DM
        response_matrix /= 2

    if flux_norm_coeff is not None:
        response_matrix /= flux_norm_coeff
    
    return response_matrix, response_cube

def make_shear_chops(camlo_ref, control_mask, shear_pix=1/2, order=3, central_diff=False, return_np=False, plot=False):
    ncamlo = camlo_ref.shape[0]
    shear_chops = xp.zeros((2, ncamlo, ncamlo))
    if central_diff:
        shear_chops_x1 = ( xcipy.ndimage.shift(camlo_ref, (0,shear_pix), order=order))
        shear_chops_x2 = ( xcipy.ndimage.shift(camlo_ref, (0,-shear_pix), order=order)) 
        shear_chops[0] = ( shear_chops_x1 - shear_chops_x2 ) / (2*shear_pix)

        shear_chops_y1 = ( xcipy.ndimage.shift(camlo_ref, (shear_pix,0), order=order) )
        shear_chops_y2 = ( xcipy.ndimage.shift(camlo_ref, (-shear_pix,0), order=order))
        shear_chops[1] = ( shear_chops_y1 - shear_chops_y2 ) / (2*shear_pix)

    else:
        shear_chops_x = ( xcipy.ndimage.shift(copy.copy(camlo_ref), (0,shear_pix), order=order))
        shear_chops_y = ( xcipy.ndimage.shift(copy.copy(camlo_ref), (shear_pix,0), order=order))

        shear_chops[0] = ( shear_chops_x - camlo_ref ) / shear_pix
        shear_chops[1] = ( shear_chops_y - camlo_ref ) / shear_pix
    shear_chops[:] *= control_mask
    shear_responses = shear_chops[:, control_mask].T
    if plot: 
        utils.imshow([shear_chops[0], shear_chops[1]])

    if return_np: 
        return ensure_np_array(shear_responses), ensure_np_array(shear_chops)
    return shear_responses, shear_chops

def plot_responses(
        dm_modes, 
        response_cube, 
        figsize=(25,5),
        dpi=125,
        hspace=0.0,
        wspace=-0.05,
        title=None,
        title_fs=14,
    ):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(2, 10, figure=fig)
    fig.suptitle(title, fontsize=title_fs)

    for i in range(10):
        mode = ensure_np_array(dm_modes[i])
        response = ensure_np_array(response_cube[i])

        ax = fig.add_subplot(gs[0, i])
        ax.imshow(mode, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[1, i])
        ax.imshow(response, cmap='magma',)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(hspace=hspace, wspace=wspace)

def compute_without_fsm_ff_offset(
        CAMLO_STREAM,
        DM_STREAM,
        LLOWFSC_REF_STREAM,
        LLOWFSC_GAINS_STREAM,
        OFFSET_STREAMS,
        P, 
        llowfsc_mask, 
        dm_modes, 
        dark_im,
        leakage=0.0,
    ):
    camlo_im = (CAMLO_STREAM.grab_after(1, 0)[0] - dark_im)
    camlo_im /= camlo_im[llowfsc_mask].sum()
    del_im = camlo_im - LLOWFSC_REF_STREAM.grab_latest()
    
    recon_coeff = 1e6*P.dot(del_im[llowfsc_mask])
    ff_offsets = np.sum([OFFSET_STREAM.grab_latest()[0] for OFFSET_STREAM in OFFSET_STREAMS], axis=0)
    coeff_with_offset = recon_coeff - ff_offsets
    modal_coeff = - LLOWFSC_GAINS_STREAM.grab_latest()[0] * coeff_with_offset[:]

    del_dm_coeff = modal_coeff[:]
    del_dm_command = np.sum( del_dm_coeff[:, None, None] * dm_modes, axis=0)
    total_lo_dm = (1 - leakage) * DM_STREAM.grab_latest() + del_dm_command
    DM_STREAM.write(total_lo_dm)

    return

def compute_without_fsm_ref_offset(
        CAMLO_STREAM,
        DM_STREAM,
        LLOWFSC_REF_STREAM,
        LLOWFSC_GAINS_STREAM,
        REF_OFFSET_STREAM,
        P, 
        llowfsc_mask, 
        dm_modes, 
        dark_im,
        leakage=0.0,
    ):
    camlo_im = (CAMLO_STREAM.grab_after(1, 0)[0] - dark_im)
    camlo_im /= camlo_im[llowfsc_mask].sum()
    del_im = camlo_im - LLOWFSC_REF_STREAM.grab_latest() - REF_OFFSET_STREAM.grab_latest()
    
    recon_coeff = 1e6*P.dot(del_im[llowfsc_mask])
    coeff_with_offset = recon_coeff
    modal_coeff = - LLOWFSC_GAINS_STREAM.grab_latest()[0] * coeff_with_offset[:]

    del_dm_coeff = modal_coeff[:]
    del_dm_command = np.sum( del_dm_coeff[:, None, None] * dm_modes, axis=0)
    total_lo_dm = (1 - leakage) * DM_STREAM.grab_latest() + del_dm_command
    DM_STREAM.write(total_lo_dm)

    return

def compute_with_fsm_ff_offset(
        CAMLO_STREAM,
        FSM_STREAM,
        DM_STREAM,
        LLOWFSC_REF_STREAM,
        LLOWFSC_GAINS_STREAM,
        OFFSET_STREAMS,
        P,
        Nz_modes,
        llowfsc_mask, 
        dm_modes, 
        fsm_beam_diam,
        dark_im,
        leakage=0.0,
    ):
    camlo_im = CAMLO_STREAM.grab_after(1, 0)[0] - dark_im
    camlo_im /= camlo_im[llowfsc_mask].sum()
    del_im = camlo_im - LLOWFSC_REF_STREAM.grab_latest()
    
    recon_coeff = P.dot(del_im[llowfsc_mask])
    zer_coeff = recon_coeff[:Nz_modes]
    ff_offsets = np.sum([OFFSET_STREAM.grab_latest()[0] for OFFSET_STREAM in OFFSET_STREAMS], axis=0) / 1e6
    coeff_with_offset = zer_coeff - ff_offsets
    modal_coeff = - LLOWFSC_GAINS_STREAM.grab_latest()[0] * coeff_with_offset

    del_fsm_coeff = modal_coeff[:2]
    del_fsm_as = utils.tt_rms_to_as(del_fsm_coeff, fsm_beam_diam)
    del_fsm_command = np.array([0, del_fsm_as[0], del_fsm_as[1]])
    total_fsm_command = (1 - leakage) * FSM_STREAM.grab_latest() + del_fsm_command
    FSM_STREAM.write(total_fsm_command)

    del_dm_coeff = modal_coeff[2:]
    del_dm_command = np.sum( del_dm_coeff[:, None, None] * dm_modes, axis=0)
    total_lo_dm = (1 - leakage) * DM_STREAM.grab_latest() + del_dm_command * 1e6
    DM_STREAM.write(total_lo_dm)

    return

def update_dm_ff_offset(
        DM_STREAM,
        DM_OFFSET_STREAM,
        z_pinv,
        factor=2,
    ):

    dm_offset_coeff = factor * z_pinv.dot(DM_STREAM.grab_latest().ravel()) # DM command should already be in terms of microns
    DM_OFFSET_STREAM.write( dm_offset_coeff[:,None].T )


