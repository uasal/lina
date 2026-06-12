from .math_module import xp, xcipy, ensure_np_array
from lina import utils, pwp

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
        'commands':[],
        'del_commands':[],
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
        # fp_shift=None,
        normalize_metric_fun=None,
        normalize_metric_params=None,
        verbose=False,
        plot_current=True,
        plot_all=False,
        vmin=1e-10,  
    ):
    """_summary_

    Args:
        aefc_data (dict): 
            Dictionary of all the corresponding data for this particular aEFC run. Dictionary contains 
            history of all previously obtained measurements and DM commands. 
        take_im_fun (callable): 
            Function that returns the image of the coronagraph.  
        take_im_params (dict): 
            Dictionary of additional parameters needed for the take_im_fun method.
        set_dm_fun (callable): 
            Function that applies the DM command to the coronagraph. First argument of 
            this function must be the DM command that will be applied. 
        set_dm_params (dict): 
            Dictionary of additional parameters needed for the set_dm_fun method.
        estimate_ef_fun (callable): 
            The function used to estimate the electric field on each iteration of aEFC. Typically this
            will be PWP or SCC. 
        estimate_ef_params (dict): 
            Dictionary of additional parameters needed for the estimate_ef_fun method. 
        M (control_models.MODEL): 
            The control model instance used to compute the new DM command. 
        val_and_grad (callable): 
            Function that returns the EFC cost-function value and gradient with respect
            to the DM actuators. Used to perform the optimization. 
        wfs_mask (ndarray):
            Binary mask defining the region in the focal plane to control.
        dm_mask (ndarray): 
            Binary array definiing the active actuators of the DM. 
        reg_cond (float, optional): 
            Regularization value for the EFC cost-function. Defaults to 1e-2.
        bfgs_tol (float, optional): 
            Tolerance for the L-BFGS-B optimization. Once the difference between two consecutive 
            optimization iterations is less than this value, optimization is completed. Defaults to 1e-3.
        bfgs_opts (dict, optional): 
            Additional dictionary of options for L-BFGS-B optimization. Defaults to None.
        num_iterations (int, optional): 
            Number of iterations to perfomr with given parameters. Defaults to 3.
        gain (float, optional): 
            Loop gain applied to each computed DM command. Defaults to 1.0.
        leakage (float, optional): 
            Leakage specifiy how much of the previous commands to remove. Defaults to 0.0.
        normalize_metric_fun (callable, optional): 
            Function that normalizes the metric image used to evaluate current contrast. If take_im_fun
            automatically returns normalized intensity images, this is not needed. Defaults to None.
        normalize_metric_params (dict, optional): 
            Dictionary of additional parameters needed for the normalize_metric_fun method. Defaults to None.
        verbose (bool, optional): 
            Option specifying if the control model propagations should use print information. Defaults to False.
        plot_current (bool, optional): 
            Plots the results of the current iteration. Defaults to True.
        plot_all (bool, optional): 
            Plots the results of all iterations performed during this round of iEFC. Defaults to False.
        vmin (float, optional): 
            Minimum contrast value to display on the plots. Defaults to 1e-9.

    Returns:
        dict: 
            Dictionary of aEFC data appended with arrays from most recent iterations. 
    """

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
        contrast = utils.compute_contrast(metric_im_ni, wfs_mask)

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


def run_mw(
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
        wfs_waves,
        reg_cond=1e-2,
        bfgs_tol=1e-3,
        bfgs_opts=None,
        num_iterations=3, 
        gain=1.0, 
        leakage=0.0, 
        weights=None, 
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

    rmad_vars = {
        'wfs_mask':wfs_mask,
        'wfs_waves':wfs_waves,
        'r_cond':reg_cond, 
        'weights':weights,
    }

    del_command = xp.zeros(dm_mask.shape) # array to fill with actuator solutions
    for i in range(num_iterations):
        print(f'Running iteration {starting_itr+i:d}')

        current_acts = total_command[dm_mask]

        E_abs = estimate_ef_fun(**estimate_ef_params)

        E_FP_NOMs = M.forward_mw(current_acts, wfs_waves, use_vortex=True, return_ints=False)

        rmad_vars.update({
            'current_acts': current_acts,
            'E_abs': E_abs,
            'E_FP_NOMs': E_FP_NOMs,
        })

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
        contrast = utils.compute_contrast(metric_im_ni, wfs_mask)

        aefc_data['raw_images'].append(copy.copy(metric_im))
        aefc_data['ni_images'].append(copy.copy(metric_im_ni))
        aefc_data['contrasts'].append(contrast)
        aefc_data['efields'].append(copy.copy(E_abs))
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




