from .math_module import xp, xcipy, ensure_np_array
from lina import utils, coro_utils, pwp

import numpy as np
from scipy.optimize import minimize
import time
import copy

import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec

def run(
        efc_data,
        CAMSCI_STREAM,
        DM_STREAM, 
        im_params,
        ref_psf_params,
        NFRAMES,
        dark_im,
        M, 
        val_and_grad,
        control_mask,
        dm_mask,
        pwp_params=None,
        Nitr=3, 
        reg_cond=1e-2,
        bfgs_tol=1e-3,
        bfgs_opts=None,
        gain=1.0, 
        leakage=0.0, 
        vmin=1e-9,
        verbose=False,  
        delay=0.05,
    ):

    starting_itr = len(efc_data['images'])

    del_command = np.zeros(DM_STREAM.shape) # array to fill with actuator solutions
    for i in range(Nitr):
        print(f'Running iteration {starting_itr+i:d}')

        current_command = DM_STREAM.grab_latest() / 1e6
        current_acts = current_command[dm_mask]

        E_FP_NOM = M.forward(current_acts, M.wavelength_c, use_vortex=True, return_ints=False)
        pwp_params.update({'E_FP_NOM':E_FP_NOM})

        E_ab, _ = pwp.run_with_model(
            CAMSCI_STREAM,
            DM_STREAM, 
            im_params,
            ref_psf_params,
            M,
            **pwp_params,
        )

        rmad_vars= {
            'E_ab': E_ab,
            'current_acts': current_acts,
            'E_FP_NOM': E_FP_NOM,
            'control_mask': control_mask,
            'wavelength':M.wavelength_c,
            'r_cond': reg_cond, 
        }

        res = minimize(
            val_and_grad, 
            jac=True, 
            x0=np.zeros(M.Nacts), # initial guess is always just zeros
            args=(M, rmad_vars, verbose, 0), 
            method='L-BFGS-B',
            tol=bfgs_tol,
            options=bfgs_opts,
        )

        del_acts = gain * res.x
        del_command[dm_mask] = del_acts
        total_command = (1 - leakage) * current_command + del_command
        DM_STREAM.write(total_command*1e6)
        time.sleep(delay)

        metric_im = np.mean(CAMSCI_STREAM.grab_many(NFRAMES), axis=0)
        metric_im_ni = coro_utils.normalize_coro_im(metric_im, im_params, ref_psf_params, dark_im=dark_im)
        mean_ni = coro_utils.compute_contrast(metric_im_ni, control_mask)

        efc_data['images'].append(copy.copy(metric_im_ni))
        efc_data['contrasts'].append(mean_ni)
        efc_data['efields'].append(copy.copy(E_ab))
        efc_data['commands'].append(copy.copy(total_command))
        efc_data['del_commands'].append(copy.copy(del_command))
        efc_data['bfgs_tols'].append(bfgs_tol)
        efc_data['reg_conds'].append(reg_cond)

        utils.imshow(
            [del_command, total_command, metric_im_ni],
            titles=['New Command', 'Total Command', f'Metric Image\nContrast = {mean_ni:.2e}'],
            norms=[None, None, LogNorm(1e-10)],
            cmaps=['viridis', 'viridis', 'magma'],
        )

    return efc_data

