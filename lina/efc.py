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

def compute_jacobian(
        model, 
        control_mask, 
        amp=1e-9, 
        current_acts=None, 
    ):

    current_acts = xp.zeros(model.Nacts) if current_acts is None else xp.array(current_acts)    

    Nmask = int(control_mask.sum())
    jac = xp.zeros((2*Nmask, model.Nacts))

    start = time.time()
    for i in range(model.Nacts):
        act_poke = xp.zeros(model.Nacts)
        act_poke[i] = amp

        E_pos = model.forward(current_acts + act_poke, model.wavelength, use_vortex=1, )
        E_neg = model.forward(current_acts - act_poke, model.wavelength, use_vortex=1, )
        response = ( E_pos - E_neg ) / (2*amp)

        jac[::2, i] = response.real[control_mask]
        jac[1::2, i] = response.imag[control_mask]

        print(f"\tCalibrated mode {i+1:d}/{model.Nacts:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")

    return jac

def run(
        efc_data,
        CAMSCI_STREAM,
        DM_STREAM, 
        im_params,
        ref_psf_params,
        NFRAMES,
        dark_im,
        control_mask,
        dm_mask,
        control_matrix,
        pwp_params=None,
        Nitr=3, 
        gain=1.0, 
        leakage=0.0,
        delay=0.05,
    ):
    
    starting_itr = len(efc_data['images'])
    Nmask = int(control_mask.sum())

    del_command = np.zeros(DM_STREAM.shape) # array to fill with actuator solutions
    E_ab_vec = xp.zeros((2*Nmask))
    for i in range(Nitr):
        print(f'Running iteration {starting_itr+i:d}')

        current_command = DM_STREAM.grab_latest() / 1e6

        E_ab_est, E_ab_est_vec = pwp.run_with_jacobian(
            CAMSCI_STREAM,
            DM_STREAM, 
            im_params,
            ref_psf_params,
            **pwp_params,
        )
        E_ab_vec[::2] = xp.real(E_ab_est_vec)
        E_ab_vec[1::2] = xp.imag(E_ab_est_vec)

        del_acts = - gain * control_matrix.dot(E_ab_vec)
        del_command[dm_mask] = ensure_np_array(del_acts)
        total_command = (1 - leakage) * current_command + del_command
        DM_STREAM.write(total_command*1e6)
        time.sleep(delay)

        metric_im = np.mean(CAMSCI_STREAM.grab_many(NFRAMES), axis=0)
        metric_im_ni = coro_utils.normalize_coro_im(metric_im, im_params, ref_psf_params, dark_im=dark_im)
        mean_ni = coro_utils.compute_contrast(metric_im_ni, control_mask)

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

def compute_jacobian_bb(
        models,
        control_mask, 
        amp=1e-9, 
        current_acts=None, 
    ):

    Nwaves = len(models)
    Nmask = int(control_mask.sum())
    jac = xp.zeros((Nwaves * 2*Nmask, models[0].Nacts))
    mono_jacs = xp.zeros((Nwaves, 2*Nmask, models[0].Nacts))
    for i in range(Nwaves):
        mono_jac = compute_jacobian(
            models[i],
            control_mask,
            amp=amp, 
            current_acts=current_acts,
        )

        mono_jacs[i] = mono_jac
        jac[i*2*Nmask:(i+1)*2*Nmask] = mono_jac

    return jac, mono_jacs


