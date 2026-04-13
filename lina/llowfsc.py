from .math_module import xp, xcipy, ensure_np_array
from lina import rt_utils, utils

import numpy as np
import astropy.units as u
import copy
from IPython.display import display, clear_output
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

def acquire_ref(
        take_im_fun,
        take_im_params,
        wfs_mask,
        camlo_dark=0.0,
        flux_norm=True,
    ):

    camlo_ref_im = take_im_fun(**take_im_params)

    camlo_ref_im_ds = camlo_ref_im - camlo_dark

    camlo_ref_im_ds *= wfs_mask

    if flux_norm:
        flux_norm_coeff = camlo_ref_im_ds[wfs_mask].sum()
        flux_norm_ref_im = camlo_ref_im_ds / flux_norm_coeff
        return flux_norm_ref_im, flux_norm_coeff
    
    return camlo_ref_im_ds

def calibrate_dm_modes(
        take_im_fun,
        take_im_params,
        set_dm_fun,
        set_dm_params,
        dm_modes, 
        wfs_mask, 
        amp=2e-9,
        base_command=None, 
        flux_norm_coeff=None,
        include_factor_2=False,
        plot=False,
    ):
    
    Nmask = int(wfs_mask.sum())
    Nmodes = dm_modes.shape[0]
    Nact = dm_modes.shape[1]
    Ncamlo = wfs_mask.shape[0]

    if base_command is None: base_command = np.zeros((Nact, Nact))

    response_matrix = np.zeros((Nmask, Nmodes))
    response_cube = np.zeros((Nmodes, Ncamlo, Ncamlo))
    
    start = time.time()
    for i in range(Nmodes):
        dm_mode = amp*dm_modes[i]

        set_dm_fun(base_command + dm_mode, **set_dm_params)
        im_pos = take_im_fun(**take_im_params)

        set_dm_fun(base_command - dm_mode, **set_dm_params)
        im_neg = take_im_fun(**take_im_params)

        set_dm_fun(base_command, **set_dm_params)

        diff = im_pos - im_neg
        response_cube[i] = copy.copy(diff) / (2 * amp)
        response_matrix[:,i] = copy.copy(diff)[wfs_mask] / (2 * amp)
        
        if plot: 
            print(f"Calibrated mode {i+1:d}/{Nmodes:d} in {time.time()-start:.3f}s", end='')
            utils.imshow(
                [dm_mode, im_pos, diff], 
                titles=[f'Mode {i+1}', 'Positive Chop Image', 'Difference'], 
                cmaps=['viridis'],
            )
        else:
            print(f"\tCalibrated mode {i+1:d}/{Nmodes:d} in {time.time()-start:.3f}s", end='')
            print("\r", end="")

    if include_factor_2: # in case you want to account for reflection off your FSM/DM
        response_matrix /= 2

    if flux_norm_coeff is not None:
        response_matrix /= flux_norm_coeff

    return response_matrix, response_cube

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

def reconstruct(
        camlo_im, 
        ref_im, 
        wfs_mask,
        control_matrix,
        dark_im=0.0,
        # which_modes='tt',
        modes=(0,10),
        flux_norm=True,
        return_del_im=False,
    ):

    camlo_im_dark_sub = camlo_im - dark_im
    camlo_im_flux_norm = camlo_im_dark_sub / camlo_im_dark_sub[wfs_mask].sum() if flux_norm else camlo_im_dark_sub
    del_im = camlo_im_flux_norm - ref_im

    coeff = control_matrix[modes[0]:modes[1]].dot(del_im[wfs_mask])

    return coeff

def run(
        take_im_fun,
        take_im_params,
        set_dm_fun,
        set_dm_params,
        get_dm_fun,
        get_dm_params,
        get_gains,
        get_gains_params,
        ref_im,
        control_matrix,
        dm_modes,
        wfs_mask,
        dark_im=0.0,
        get_zpo=None,
        get_zpo_params={},
        get_ffo=None,
        get_ffo_params={},
    ):

    camlo_im = take_im_fun(**take_im_params)

    zpo = get_zpo(**get_zpo_params) if get_zpo is not None else 0.0
    recon_coeff = reconstruct(
        camlo_im, 
        ref_im + zpo, 
        wfs_mask,
        control_matrix,
        dark_im=dark_im,
        modes=(0,10),
    )
    ffo = get_ffo(**get_ffo_params) if get_ffo is not None else 0.0
    recon_coeff -= ffo

    modal_coeff = - get_gains(**get_gains_params) * recon_coeff

    del_dm_command = np.sum(modal_coeff[:, None, None] * dm_modes, axis=0)

    total_dm_command = get_dm_fun(**get_dm_params) + del_dm_command

    set_dm_fun(total_dm_command, **set_dm_params)

import threading
class Process(threading.Timer):  
    def run(self):
        while not self.finished.wait(self.interval):  
            self.function(*self.args, **self.kwargs)

# process = Process(0.1, print, ['Repeating']) 
# process.start()
# process.cancel()

def compute_zpo(
        DM_STREAMS,
        dm_mask,
        wfs_mask,
        response_matrix,
        dm_modal_matrix,
        ZPO_STREAM,
    ):
    zpo = np.zeros((wfs_mask.shape[0], wfs_mask.shape[1]))
    for i in range(len(DM_STREAMS)):
        zpo[wfs_mask] += response_matrix.dot( dm_modal_matrix.dot(DM_STREAMS[i].grab_latest()[dm_mask]) )

    ZPO_STREAM.write(zpo)
    return zpo

