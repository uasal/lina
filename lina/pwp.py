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
        CAMSCI_STREAM,
        DM_STREAM, 
        im_params,
        ref_psf_params,
        NFRAMES,
        control_mask, 
        dm_mask,
        probes, 
        probe_amp, 
        jacobian=None,
        model=None,
        wavelength=None,
        E_FP_NOM=None,
        fp_shift=None,
        reg_cond=1e-3, 
        gain=1,
        plot=False,
        plot_est=False, 
        return_all=False,
        delay=0.05,
    ):
    
    Nmask = int(control_mask.sum())
    Nprobes = probes.shape[0]

    current_command = DM_STREAM.grab_latest() / 1e6
    current_acts = current_command[dm_mask]

    all_ims = []
    diff_ims = []
    E_probes = []
    for i in range(Nprobes):
        dm_probe = probe_amp*probes[i]

        DM_STREAM.write( (current_command + dm_probe)*1e6)
        time.sleep(delay)
        im_pos = np.mean(CAMSCI_STREAM.grab_many(NFRAMES), axis=0)

        DM_STREAM.write( (current_command - dm_probe)*1e6)
        time.sleep(delay)
        im_neg = np.mean(CAMSCI_STREAM.grab_many(NFRAMES), axis=0)

        diff_im = im_pos - im_neg
        diff_im_ni = coro_utils.normalize_coro_im(diff_im, im_params, ref_psf_params, dark_im=0.0)
        if fp_shift is not None:
            scipy.ndimage.shift(diff_im_ni, (fp_shift[1], fp_shift[0]), order=0)

        probe_acts = probe_amp*probes[i][dm_mask]

        if jacobian is not None:
            E_probe_vec = xp.array(jacobian.dot(xp.array(probe_acts)))
            E_probe = xp.zeros(CAMSCI_STREAM.shape, dtype=xp.complex128)
            E_probe[control_mask] = E_probe_vec[::2] + 1j*E_probe_vec[1::2]
        elif model is not None:
            if i==0 and E_FP_NOM is None: 
                E_FP_NOM = model.forward(current_acts, wavelength, use_vortex=True)
            E_with_probe = model.forward(current_acts + probe_acts, wavelength, use_vortex=True)
            E_probe = E_with_probe - E_FP_NOM
        else:
            raise ValueError('Must provide either a Jacobian or a forward model to estimate E-field response of probes.')

        all_ims.append([im_pos, im_neg])
        diff_ims.append(diff_im_ni)
        E_probes.append(E_probe)

        if plot:
            utils.imshow(
                [dm_probe, im_pos, diff_im, diff_im_ni], 
                titles=['DM Probe', 'Positive Chop Image', 'Difference Image', 'Normalized Difference Image'],
                norms=[None, LogNorm(np.max(im_pos)/1e4), None, None],
                cmaps=['viridis', 'magma', 'magma', 'magma'],
                figsize=(18, 8),
                wspace=0.3,
                xticks=[[], [], [], []],
                yticks=[[], [], [], []],
            )

    DM_STREAM.write( current_command*1e6 )
    all_ims = xp.array(all_ims)
    diff_ims = xp.array(diff_ims)
    E_probes = xp.array(E_probes)

    # Use batch process to estimate each pixel individually
    E_est = xp.zeros(Nmask, dtype=xp.complex128)
    for i in range(Nmask):
        delI = diff_ims[:, control_mask][:, i]
        H = 4*xp.array(
            [E_probes[:, control_mask][:, i].real, 
             E_probes[:, control_mask][:, i].imag]
        ).T # Dimensions are 2 X N_probes
        
        Hinv = xp.linalg.pinv(H.T @ H, reg_cond) @ H.T
        # print(H.shape, Hinv.shape, delI.shape)
        est = Hinv.dot(delI)

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = xp.zeros(CAMSCI_STREAM.shape, dtype=xp.complex128)
    E_est_2d[control_mask] = gain * E_est

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
        return E_est_2d, E_est, E_probes, diff_ims
    else:
        return E_est_2d, E_est

