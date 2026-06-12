from .math_module import xp, xcipy, ensure_np_array
from lina import utils, pwp

import numpy as np
import time
import copy
from IPython.display import display, clear_output

import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize, CenteredNorm
from matplotlib.gridspec import GridSpec

def calibrate(
        compute_ef_fun,
        compute_ef_params,
        dm_mask,
        wfs_mask, 
        amp=1e-9, 
        current_command=None, 
        plot_response_map=False,
    ):
    """
    This function computes a Jacobian for EFC using a model of the coronagraph. 

    Args:
        compute_ef_fun (callable):
            Function that computes the electric field at the focal plane of the coronagraph model.
            First argument of this function is the DM command that will be applied for the 
            electric field that is computed. 
        compute_ef_params (dict): 
            Dictionary of additional parameters provided to the compute_ef_fun method. 
        dm_mask (ndarray): 
            Binary array definiing the active actuators of the DM. 
        wfs_mask (ndarray):
            Binary array defining the region of the focal plane for which the electric field 
            response is stored within the Jacobian. 
        amp (float, optional): 
            Amplitude of the actuator pokes used to compute the electric field response 
            in units of meters. Defaults to 1e-9.
        current_command (ndarray, optional): 
            Base command the DM pokes are added on top of for calibration. Defaults to None.

    Returns:
        jac (ndarray):
            The jacobian (or response matrix) computed by the instrument model.
    """

    Nacts = int(dm_mask.sum())
    current_command = xp.zeros(dm_mask.shape) if current_command is None else xp.array(current_command)    

    Nmask = int(wfs_mask.sum())
    jac = xp.zeros((2*Nmask, Nacts))

    start = time.time()
    for i in range(Nacts):
        act_poke = xp.zeros(Nacts)
        act_poke[i] = amp
        poke_command = xp.zeros(dm_mask.shape)
        poke_command[dm_mask] = act_poke

        # print(current_command)
        E_pos = compute_ef_fun(current_command + poke_command, **compute_ef_params)
        E_neg = compute_ef_fun(current_command - poke_command, **compute_ef_params)
        response = ( E_pos - E_neg ) / (2*amp)

        jac[::2, i] = response.real[wfs_mask]
        jac[1::2, i] = response.imag[wfs_mask]

        print(f"\tCalibrated mode {i+1:d}/{Nacts:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")

    if plot_response_map:
        dm_response_rms = xp.sqrt(xp.mean(xp.square(xp.abs(jac)), axis=0))
        dm_response_map = xp.zeros(dm_mask.shape)
        dm_response_map[dm_mask] = dm_response_rms/dm_response_rms.max()
        utils.imshow(
            [dm_response_map],
            norms=[LogNorm(1e-2)],
        )

    return jac

def init_data(
        wfs_mask=None, 
        contrast0=None,
        ni_im0=None,
    ):
    efc_data = {
        'raw_images':[],
        'ni_images':[],
        'efields':[],
        'contrasts':[],
        'commands':[],
        'del_commands':[],
        'reg_conds':[],
        'wfs_mask':wfs_mask,
        'ni_im0':ni_im0,
        'contrast0':contrast0,
    }
    return efc_data

def run(efc_data,
        take_im_fun,
        take_im_params,
        set_dm_fun,
        set_dm_params,
        estimate_ef_fun,
        estimate_ef_params,
        wfs_mask,
        dm_mask,
        response_matrix,
        reg_cond,
        wfs_mask_mw = None,
        normalize_metric_fun=None,
        normalize_metric_params=None,
        num_iterations=3, 
        gain=1.0, 
        leakage=0.0,
        plot_current=True,
        plot_all=False,
        vmin=1e-10,
        vmax=1e-5,
    ):
    """
    Run EFC for a set amount of iterations.

    Args:
        efc_data (dict): 
            Dictionary of all the corresponding data for this particular EFC run. Dictionary contains 
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
            The function used to estimate the electric field on each iteration of EFC. Typically this
            will be PWP or SCC. 
        estimate_ef_params (dict): 
            Dictionary of additional parameters needed for the estimate_ef_fun method. 
        wfs_mask (ndarray):
            Binary mask defining the region in the focal plane to control.
        dm_mask (ndarray): 
            Binary array definiing the active actuators of the DM. 
        control_matrix (ndarray): 
            Pseudo-inverted response matrix for the region of interest specified 
            by the wfs_mask. 
        wfs_mask_mw (ndarray):
            Binary mask defining the region in the focal plane to control when using multi band WFS.
        normalize_metric_fun (callable, optional): 
            Function that normalizes the metric image used to evaluate current contrast. If take_im_fun
            automatically returns normalized intensity images, this is not needed. Defaults to None.
        normalize_metric_params (dict, optional): 
            Dictionary of additional parameters needed for the normalize_metric_fun method. Defaults to None.
        num_iterations (int, optional): 
            Number of iterati9ons to perform iEFC with these specific parameters. Defaults to 3.
        gain (float, optional): 
            Loop gain applied to each computed DM command. Defaults to 1.0.
        leakage (float, optional): 
            Leakage specifiy how much of the previous commands to remove. Defaults to 0.0.
        plot_current (bool, optional): 
            Plots the results of the current iteration. Defaults to True.
        plot_all (bool, optional): 
            Plots the results of all iterations performed during this round of iEFC. Defaults to False.
        vmin (float, optional): 
            Minimum contrast value to display on the plots. Defaults to 1e-10.
        vmax (float, optional): 
            Maximum contrast value to display on the plots. Defaults to 1e-5.

    Returns:
        efc_data (dict): 
            Dictionary of EFC data appended with the results of the new iterations performed. 
    """

    if wfs_mask_mw is None:    # set multi wave mask to single wave mask if not provided with one
        wfs_mask_mw = wfs_mask
    
    Nmask = int(wfs_mask_mw.sum())
    Nact = dm_mask.shape[0]

    starting_itr = len(efc_data['commands']) + 1
    total_command = copy.copy(efc_data['commands'][-1]) if len(efc_data['commands'])>0 else xp.zeros((Nact,Nact))

    control_matrix = utils.beta_reg(response_matrix, reg_cond)

    del_command = xp.zeros(dm_mask.shape) # array to fill with actuator solutions
    E_ab_vec = xp.zeros((2*Nmask))
    for i in range(num_iterations):
        print(f'Running iteration {starting_itr+i:d}')

        E_ab = estimate_ef_fun(**estimate_ef_params)
        E_ab_vec[::2] = xp.real(E_ab[wfs_mask_mw])
        E_ab_vec[1::2] = xp.imag(E_ab[wfs_mask_mw])

        del_acts = - gain * control_matrix.dot(E_ab_vec)
        del_command[dm_mask] = del_acts
        total_command = (1 - leakage) * total_command + del_command
        
        set_dm_fun(total_command, **set_dm_params)

        metric_im = take_im_fun(**take_im_params)
        metric_im_ni = metric_im if normalize_metric_fun is None else normalize_metric_fun(metric_im, **normalize_metric_params)
        contrast = utils.compute_contrast(metric_im_ni, wfs_mask)

        print(f'\tContrast = {contrast:.3e}.')

        efc_data['raw_images'].append(copy.copy(metric_im))
        efc_data['ni_images'].append(copy.copy(metric_im_ni))
        efc_data['contrasts'].append(copy.copy(contrast))
        efc_data['efields'].append(copy.copy(E_ab))
        efc_data['commands'].append(copy.copy(total_command))
        efc_data['del_commands'].append(copy.copy(del_command))
        efc_data['reg_conds'].append(copy.copy(reg_cond))

        if plot_current: 
            if not plot_all: clear_output(wait=True)
            utils.imshow(
                [del_command, total_command, metric_im_ni], 
                titles=[f'Iteration {starting_itr + i:d}: $\delta$DM', 
                        'Total DM Command', 
                        f'Normalized Image\nMean Contrast = {contrast:.3e}'],
                cmaps=['viridis', 'viridis', 'magma'],
                pxscls=[None, None, None],
                norms=[CenteredNorm(), None, LogNorm(vmin, vmax)],
            )

    return efc_data




