from .math_module import xp, xcipy, ensure_np_array
from lina import utils, coro_utils, pwp

import numpy as np
import scipy
from scipy.optimize import minimize
import time
import copy
from IPython.display import display, clear_output

import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize, CenteredNorm
from matplotlib.gridspec import GridSpec

def init_data():
    aefc_data = {
        'raw_images':[],
        'ni_images':[],
        'efields':[],
        'contrasts':[],
        'commands':[],
        'del_commands':[],
        'bfgs_tols':[],
        'reg_conds':[],
    }
    return aefc_data

def run(
        aefc_data,
        take_im_fun,
        take_im_params,
        set_dm_fun,
        set_dm_params,
        estimate_ef_fun,
        estimate_ef_params,
        M, 
        val_and_grad,
        wfs_mask,
        dm_mask,
        reg_cond=1e-2,
        bfgs_tol=1e-3,
        bfgs_opts=None,
        num_iterations=3, 
        gain=1.0, 
        leakage=0.0, 
        fp_shift=None,
        normalize_metric_fun=None,
        normalize_metric_params=None,
        verbose=False,
        plot_current=True,
        plot_all=False,
        vmin=1e-10,  
    ):

    Nact = dm_mask.shape[0]

    starting_itr = len(aefc_data['commands']) + 1
    total_command = copy.copy(aefc_data['commands'][-1]) if len(aefc_data['commands'])>0 else xp.zeros((Nact,Nact))

    del_command = xp.zeros(dm_mask.shape) # array to fill with actuator solutions
    for i in range(num_iterations):
        print(f'Running iteration {starting_itr+i:d}')

        E_ab = estimate_ef_fun(**estimate_ef_params)

        current_acts = total_command[dm_mask]
        E_FP_NOM = M.forward(current_acts, M.wavelength_c, use_vortex=True, return_ints=False)

        rmad_vars= {
            'E_ab': E_ab,
            'current_acts': current_acts,
            'E_FP_NOM': E_FP_NOM,
            'wfs_mask': wfs_mask,
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

        del_acts = xp.array( gain * res.x )
        del_command[dm_mask] = del_acts
        total_command = (1 - leakage) * total_command + del_command
        
        set_dm_fun(total_command, **set_dm_params)

        metric_im = take_im_fun(**take_im_params)
        metric_im_ni = metric_im if normalize_metric_fun is None else normalize_metric_fun(metric_im, **normalize_metric_params)
        contrast = coro_utils.compute_contrast(metric_im_ni, wfs_mask)

        aefc_data['raw_images'].append(copy.copy(metric_im))
        aefc_data['ni_images'].append(copy.copy(metric_im_ni))
        aefc_data['contrasts'].append(contrast)
        aefc_data['efields'].append(copy.copy(E_ab))
        aefc_data['commands'].append(copy.copy(total_command))
        aefc_data['del_commands'].append(copy.copy(del_command))
        aefc_data['bfgs_tols'].append(bfgs_tol)
        aefc_data['reg_conds'].append(reg_cond)

        if plot_current: 
            if not plot_all: clear_output(wait=True)
            utils.imshow(
                [del_command, total_command, metric_im_ni], 
                titles=[f'Iteration {starting_itr + i:d}: $\delta$DM', 
                        'Total DM Command', 
                        f'Normalized Image\nMean Contrast = {contrast:.3e}'],
                cmaps=['viridis', 'viridis', 'magma'],
                pxscls=[None, None, None],
                norms=[CenteredNorm(), None, LogNorm(vmin=vmin)],
            )

    return aefc_data

