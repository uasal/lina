from lina.math_module import xp, xcipy, ensure_np_array
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

def init_data(
        wfs_mask=None, 
        contrast0=None,
        ni_im0=None,
    ):
    aefc_data = {
        'raw_images':[],
        'ni_images':[],
        'efields':[],
        'contrasts':[],
        'dm1_commands':[],
        'del_dm1_commands':[],
        'dm2_commands':[],
        'del_dm2_commands':[],
        'bfgs_tols':[],
        'reg_conds':[],
        'wfs_mask':wfs_mask,
        'ni_im0':ni_im0,
        'contrast0':contrast0,
    }
    return aefc_data

def run(
        aefc_data,
        take_im_fun,
        take_im_params,
        set_dm_fun,
        set_dm_params,
        get_dm_fun, 
        get_dm_params,
        estimate_ef_fun,
        estimate_ef_params,
        M, 
        val_and_grad,
        wfs_mask,
        dm_mask,
        # wfs_waves=None,
        reg_cond=1e-2,
        bfgs_tol=1e-3,
        bfgs_opts=None,
        num_iterations=3, 
        gain=1.0, 
        leakage=0.0,
        weights=None,  
        normalize_metric_fun=None,
        normalize_metric_params=None,
        verbose=False,
        plot_current=True,
        plot_all=False,
        vmin=1e-10,  
    ):

    Nact = dm_mask.shape[0]
    Nacts = int(dm_mask.sum())

    starting_itr = len(aefc_data['ni_images']) + 1
    total_dm1_command, total_dm2_command = get_dm_fun(**get_dm_params)
    
    # rmad_vars = {
    #     'wfs_mask':wfs_mask,
    #     'r_cond':reg_cond, 
    # }

    del_dm1_command = xp.zeros(dm_mask.shape) # array to fill with actuator solutions
    del_dm2_command = xp.zeros(dm_mask.shape)
    for i in range(num_iterations):
        print(f'Running iteration {starting_itr+i:d}')

        E_ab = estimate_ef_fun(**estimate_ef_params)

        current_acts = xp.concatenate([total_dm1_command[dm_mask], total_dm2_command[dm_mask]])
        E_FP_NOM = M.forward(current_acts, M.wavelength_c, use_vortex=True, return_ints=False)

        rmad_vars = {
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
            x0=np.zeros(2*M.Nacts), # initial guess is always just zeros
            args=(M, rmad_vars, verbose, 0), 
            method='L-BFGS-B',
            tol=bfgs_tol,
            options=bfgs_opts,
        )

        del_acts = xp.array( gain * res.x )

        del_dm1_command[dm_mask] = del_acts[:Nacts]
        total_dm1_command = (1 - leakage) * total_dm1_command + del_dm1_command

        del_dm2_command[dm_mask] = del_acts[Nacts:]
        total_dm2_command = (1 - leakage) * total_dm2_command + del_dm2_command
        
        set_dm_fun(total_dm1_command, total_dm2_command, **set_dm_params)

        metric_im = take_im_fun(**take_im_params)
        metric_im_ni = metric_im if normalize_metric_fun is None else normalize_metric_fun(metric_im, **normalize_metric_params)
        contrast = coro_utils.compute_contrast(metric_im_ni, wfs_mask)

        aefc_data['raw_images'].append(copy.copy(metric_im))
        aefc_data['ni_images'].append(copy.copy(metric_im_ni))
        aefc_data['contrasts'].append(contrast)
        aefc_data['efields'].append(copy.copy(E_ab))
        aefc_data['dm1_commands'].append(copy.copy(total_dm1_command))
        aefc_data['del_dm1_commands'].append(copy.copy(del_dm1_command))
        aefc_data['dm2_commands'].append(copy.copy(total_dm2_command))
        aefc_data['del_dm2_commands'].append(copy.copy(del_dm2_command))
        aefc_data['bfgs_tols'].append(bfgs_tol)
        aefc_data['reg_conds'].append(reg_cond)

        if plot_current: 
            if not plot_all: clear_output(wait=True)
            utils.imshow(
                [del_dm1_command, total_dm1_command], 
                titles=[
                    f'Iteration {starting_itr + i:d}: $\delta$DM1', 
                    'Total DM1 Command', 
                ],
                cmaps=['viridis', 'viridis',],
                pxscls=[None, None,],
                norms=[CenteredNorm(), None,],
            )

            utils.imshow(
                [del_dm2_command, total_dm2_command], 
                titles=[
                    f'Iteration {starting_itr + i:d}: $\delta$DM2', 
                    'Total DM2 Command', 
                ],
                cmaps=['viridis', 'viridis',],
                pxscls=[None, None,],
                norms=[CenteredNorm(), None,],
            )

            utils.imshow(
                [metric_im_ni], 
                titles=[f'Normalized Image\nMean Contrast = {contrast:.3e}',],
                norms=[LogNorm(vmin=vmin)],
            )


    return aefc_data


def run_mw(
        aefc_data,
        take_im_fun,
        take_im_params,
        set_dm_fun,
        set_dm_params,
        get_dm_fun, 
        get_dm_params,
        estimate_ef_fun,
        estimate_ef_params,
        M, 
        val_and_grad,
        wfs_mask,
        dm_mask,
        wfs_waves=None,
        reg_cond=1e-2,
        bfgs_tol=1e-3,
        bfgs_opts=None,
        num_iterations=3, 
        gain=1.0, 
        leakage=0.0,
        weights=None,  
        normalize_metric_fun=None,
        normalize_metric_params=None,
        verbose=False,
        plot_current=True,
        plot_all=False,
        vmin=1e-10,  
    ):

    Nact = dm_mask.shape[0]
    Nacts = int(dm_mask.sum())

    starting_itr = len(aefc_data['ni_images']) + 1
    total_dm1_command, total_dm2_command = get_dm_fun(**get_dm_params)
    
    rmad_vars = {
        'wfs_mask':wfs_mask,
        'wfs_waves':wfs_waves,
        'r_cond':reg_cond, 
        'weights':weights,
    }

    del_dm1_command = xp.zeros(dm_mask.shape) # array to fill with actuator solutions
    del_dm2_command = xp.zeros(dm_mask.shape)
    for i in range(num_iterations):
        print(f'Running iteration {starting_itr+i:d}')
        
        # For MW
        E_abs = estimate_ef_fun(**estimate_ef_params)

        current_acts = xp.concatenate([total_dm1_command[dm_mask], total_dm2_command[dm_mask]])
        E_FP_NOMs = M.forward_mw(current_acts, wfs_waves, use_vortex=True, return_ints=False)

        rmad_vars.update({
            'current_acts': current_acts,
            'E_abs': E_abs,
            'E_FP_NOMs': E_FP_NOMs,
        })

        res = minimize(
            val_and_grad, 
            jac=True, 
            x0=np.zeros(2*M.Nacts), # initial guess is always just zeros
            args=(M, rmad_vars, verbose, 0), 
            method='L-BFGS-B',
            tol=bfgs_tol,
            options=bfgs_opts,
        )

        del_acts = xp.array( gain * res.x )

        del_dm1_command[dm_mask] = del_acts[:Nacts]
        total_dm1_command = (1 - leakage) * total_dm1_command + del_dm1_command

        del_dm2_command[dm_mask] = del_acts[Nacts:]
        total_dm2_command = (1 - leakage) * total_dm2_command + del_dm2_command
        
        set_dm_fun(total_dm1_command, total_dm2_command, **set_dm_params)

        metric_im = take_im_fun(**take_im_params)
        metric_im_ni = metric_im if normalize_metric_fun is None else normalize_metric_fun(metric_im, **normalize_metric_params)
        contrast = coro_utils.compute_contrast(metric_im_ni, wfs_mask)

        aefc_data['raw_images'].append(copy.copy(metric_im))
        aefc_data['ni_images'].append(copy.copy(metric_im_ni))
        aefc_data['contrasts'].append(contrast)
        aefc_data['efields'].append(copy.copy(E_abs))
        aefc_data['dm1_commands'].append(copy.copy(total_dm1_command))
        aefc_data['del_dm1_commands'].append(copy.copy(del_dm1_command))
        aefc_data['dm2_commands'].append(copy.copy(total_dm2_command))
        aefc_data['del_dm2_commands'].append(copy.copy(del_dm2_command))
        aefc_data['bfgs_tols'].append(bfgs_tol)
        aefc_data['reg_conds'].append(reg_cond)

        if plot_current: 
            if not plot_all: clear_output(wait=True)
            utils.imshow(
                [del_dm1_command, total_dm1_command], 
                titles=[
                    f'Iteration {starting_itr + i:d}: $\delta$DM1', 
                    'Total DM1 Command', 
                ],
                cmaps=['viridis', 'viridis',],
                pxscls=[None, None,],
                norms=[CenteredNorm(), None,],
            )

            utils.imshow(
                [del_dm2_command, total_dm2_command], 
                titles=[
                    f'Iteration {starting_itr + i:d}: $\delta$DM2', 
                    'Total DM2 Command', 
                ],
                cmaps=['viridis', 'viridis',],
                pxscls=[None, None,],
                norms=[CenteredNorm(), None,],
            )

            utils.imshow(
                [metric_im_ni], 
                titles=[f'Normalized Image\nMean Contrast = {contrast:.3e}',],
                norms=[LogNorm(vmin=vmin)],
            )


    return aefc_data




