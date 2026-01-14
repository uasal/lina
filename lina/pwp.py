from .math_module import xp, xcipy, ensure_np_array
from lina import utils, coro_utils

import numpy as np
import scipy
import time
import copy

import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec

def run(
        take_im_fun,
        take_im_params,
        set_dm_fun,
        set_dm_params,
        get_dm_fun,
        get_dm_params,
        compute_probe_ef_fun,
        compute_probe_ef_params,
        wfs_mask, 
        # dm_mask,
        probe_modes, 
        probe_amp, 
        base_command=None,
        normalize_diff_fun=None,
        normalize_diff_params=None,
        # jacobian=None,
        # model=None,
        # wavelength=None,
        # E_FP_NOM=None,
        fp_shift=None,
        reg_cond=1e-3, 
        gain=1,
        plot=False,
        plot_est=False, 
        return_all=False,
    ):
    
    Nmask = int(wfs_mask.sum())
    Nprobes = probe_modes.shape[0]
    Nact = probe_modes.shape[1]

    # if base_command is None: base_command = xp.zeros((Nact, Nact))
    base_command = get_dm_fun(**get_dm_params)

    all_ims = []
    diff_ims = []
    probe_efs = []
    for i in range(Nprobes):
        probe = probe_amp*probe_modes[i]

        set_dm_fun(base_command + probe, **set_dm_params)
        im_pos = take_im_fun(**take_im_params)

        set_dm_fun(base_command - probe, **set_dm_params)
        im_neg = take_im_fun(**take_im_params)

        diff_im = im_pos - im_neg
        diff_im_ni = diff_im if normalize_diff_fun is None else normalize_diff_fun(diff_im, **normalize_diff_params)
        if fp_shift is not None:
            xcipy.ndimage.shift(diff_im_ni, (fp_shift[1], fp_shift[0]), order=0)

        probe_ef = compute_probe_ef_fun(probe, **compute_probe_ef_params)

        all_ims.append([im_pos, im_neg])
        diff_ims.append(diff_im_ni)
        probe_efs.append(probe_ef)

        if plot:
            utils.imshow(
                [probe, im_pos, diff_im_ni], 
                titles=['DM Probe', 'Positive Chop Image', 'Normalized Difference Image'],
                norms=[None, LogNorm(xp.max(im_pos)/1e4), None, None],
                cmaps=['viridis', 'magma', 'magma', 'magma'],
                # figsize=(18, 8),
                wspace=0.3,
                xticks=[[], [], [], []],
                yticks=[[], [], [], []],
            )

    set_dm_fun(base_command, **set_dm_params)

    all_ims = xp.array(all_ims)
    diff_ims = xp.array(diff_ims)
    probe_efs = xp.array(probe_efs)

    # Use batch process to estimate each pixel individually
    E_est = xp.zeros(Nmask, dtype=xp.complex128)
    for i in range(Nmask):
        delI = diff_ims[:, wfs_mask][:, i]
        H = 4*xp.array(
            [probe_efs[:, wfs_mask][:, i].real, 
             probe_efs[:, wfs_mask][:, i].imag]
        ).T # Dimensions are 2 X N_probes
        
        Hinv = xp.linalg.pinv(H.T @ H, reg_cond) @ H.T
        # print(H.shape, Hinv.shape, delI.shape)
        est = Hinv.dot(delI)

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = xp.zeros(wfs_mask.shape, dtype=xp.complex128)
    E_est_2d[wfs_mask] = gain * E_est

    if plot_est:
        I_est = xp.abs(E_est_2d)**2
        P_est = xp.angle(E_est_2d)
        utils.imshow(
            [I_est, P_est], 
            titles=['Estimated Intensity', 'Estimated Phase'],
            norms=[LogNorm(xp.max(I_est)/1e4), None],
            cmaps=['magma', 'twilight'],
        )

    if return_all:
        return E_est_2d, E_est, probe_efs, diff_ims
    else:
        return E_est_2d

