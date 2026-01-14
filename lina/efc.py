from .math_module import xp, xcipy, ensure_np_array
from lina import utils, coro_utils, pwp

import numpy as np
import time
import copy

import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec

def calibrate(
        # model, 
        compute_ef_fun,
        compute_ef_params,
        dm_mask,
        wfs_mask, 
        amp=1e-9, 
        current_command=None, 
        plot_response_map=True,
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
        current_acts (ndarray, optional): . Defaults to None.

    Returns:
        _type_: _description_
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
            [dm_response_map]
        )

    return jac

def run(efc_data,
        take_im_fun,
        take_im_params,
        set_dm_fun,
        set_dm_params,
        estimate_ef_fun,
        estimate_ef_params,
        wfs_mask,
        dm_mask,
        control_matrix,
        normalize_metric_fun=None,
        normalize_metric_params=None,
        num_iterations=3, 
        gain=1.0, 
        leakage=0.0,
    ):
    
    Nmask = int(wfs_mask.sum())

    starting_itr = len(efc_data['commands'])
    total_command = copy.copy(efc_data['commands'][-1]) if len(efc_data['commands'])>0 else xp.zeros((Nact,Nact))

    del_command = xp.zeros(dm_mask.shape) # array to fill with actuator solutions
    E_ab_vec = xp.zeros((2*Nmask))
    for i in range(num_iterations):
        print(f'Running iteration {starting_itr+i:d}')

        E_ab = estimate_ef_fun(**estimate_ef_params)
        E_ab_vec[::2] = xp.real(E_ab[wfs_mask])
        E_ab_vec[1::2] = xp.imag(E_ab[wfs_mask])

        del_acts = - gain * control_matrix.dot(E_ab_vec)
        del_command[dm_mask] = del_acts
        total_command = (1 - leakage) * total_command + del_command
        
        set_dm_fun(total_command, **set_dm_params)

        metric_im = take_im_fun(**take_im_params)
        metric_im_ni = metric_im if normalize_metric_fun is None else normalize_metric_fun(metric_im, **normalize_metric_params)
        contrast = coro_utils.compute_contrast(metric_im_ni, wfs_mask)

        efc_data['raw_images'].append(copy.copy(metric_im))
        efc_data['ni_images'].append(copy.copy(metric_im_ni))
        efc_data['contrasts'].append(copy.copy(contrast))
        efc_data['commands'].append(copy.copy(total_command))
        efc_data['del_commands'].append(copy.copy(del_command))

        utils.imshow(
            [del_command, total_command, metric_im_ni],
            titles=[f'Iteration {starting_itr + i:d}: $\delta$DM', 'Total Command', f'Metric Image\nContrast = {contrast:.2e}'],
            norms=[None, None, LogNorm(1e-10)],
            cmaps=['viridis', 'viridis', 'magma'],
        )

    return efc_data

def compute_jacobian_bb(
        models,
        wfs_mask, 
        amp=1e-9, 
        current_acts=None, 
    ):

    Nwaves = len(models)
    Nmask = int(wfs_mask.sum())
    jac = xp.zeros((Nwaves * 2*Nmask, models[0].Nacts))
    mono_jacs = xp.zeros((Nwaves, 2*Nmask, models[0].Nacts))
    for i in range(Nwaves):
        mono_jac = compute_jacobian(
            models[i],
            wfs_mask,
            amp=amp, 
            current_acts=current_acts,
        )

        mono_jacs[i] = mono_jac
        jac[i*2*Nmask:(i+1)*2*Nmask] = mono_jac

    return jac, mono_jacs


def run_bb(
        efc_data,
        CAMSCI_STREAM,
        DM_STREAM, 
        im_params,
        ref_psf_params,
        NFRAMES,
        dark_im,
        wfs_mask,
        dm_mask,
        control_matrix,
        waves,
        pwp_params=None,
        Nitr=3, 
        gain=1.0, 
        leakage=0.0,
        dm_delay=0.05,
        filter_delay=1.0,
    ):
    
    Nwaves = len(waves)
    starting_itr = len(efc_data['images'])
    Nmask = int(wfs_mask.sum())

    del_command = np.zeros(DM_STREAM.shape) # array to fill with actuator solutions
    E_ab_bb_vec = xp.zeros((Nwaves * 2*Nmask))
    for i in range(Nitr):
        print(f'Running iteration {starting_itr+i:d}')
        current_command = DM_STREAM.grab_latest() / 1e6

        for j in range(Nwaves):
            # SET FILTER WHEEL TO DESIRED POSITION
            time.sleep(filter_delay)
            E_ab_nb_vec = xp.zeros((2*Nmask))
            E_ab_est, E_ab_est_vec = pwp.run_with_jacobian(
                CAMSCI_STREAM,
                DM_STREAM, 
                im_params,
                ref_psf_params,
                **pwp_params,
            )
            E_ab_nb_vec[::2] = xp.real(E_ab_est_vec)
            E_ab_nb_vec[1::2] = xp.imag(E_ab_est_vec)

            E_ab_bb_vec[j*2*Nmask:(j+1)*2*Nmask] = E_ab_nb_vec

        del_acts = - gain * control_matrix.dot(E_ab_bb_vec)
        del_command[dm_mask] = ensure_np_array(del_acts)
        total_command = (1 - leakage) * current_command + del_command
        DM_STREAM.write(total_command*1e6)
        time.sleep(dm_delay)

        metric_im = np.mean(CAMSCI_STREAM.grab_many(NFRAMES), axis=0)
        metric_im_ni = coro_utils.normalize_coro_im(metric_im, im_params, ref_psf_params, dark_im=dark_im)
        mean_ni = coro_utils.compute_contrast(metric_im_ni, wfs_mask)

        efc_data['images'].append(copy.copy(metric_im_ni))
        efc_data['contrasts'].append(mean_ni)
        efc_data['efields'].append(copy.copy(E_ab_est))
        efc_data['commands'].append(copy.copy(total_command))
        efc_data['del_commands'].append(copy.copy(del_command))

        utils.imshow(
            [del_command, total_command, metric_im_ni],
            titles=['New Command', 'Total Command', f'Metric Image\nContrast = {mean_ni:.2e}'],
            norms=[None, None, LogNorm(1e-10)],
            cmaps=['viridis', 'viridis', 'magma'],
        )

    return efc_data


