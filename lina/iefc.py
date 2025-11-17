from .math_module import xp, xcipy, ensure_np_array
from esc_llowfsc_sim import utils, coro_utils

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize, CenteredNorm

def take_measurement(
        CAMSCI_STREAM, 
        NCAMSCI,
        DM_STREAM, 
        im_params,
        ref_psf_params,
        # dark_im,
        probe_modes,
        probe_amplitude,  
        delay=0.01,
        plot=False
    ):
    
    Ncamsci = CAMSCI_STREAM.shape[0]
    Nprobes = probe_modes.shape[0]

    current_command = DM_STREAM.grab_latest() / 1e6
    
    all_ims = []
    probed_responses = []
    ims = []
    for i in range(Nprobes):
        probe = ensure_np_array(probe_amplitude * probe_modes[i])

        DM_STREAM.write( (current_command + probe)*1e6)
        time.sleep(delay)
        im_pos = np.mean(CAMSCI_STREAM.grab_many(NCAMSCI), axis=0)

        DM_STREAM.write( (current_command - probe)*1e6)
        time.sleep(delay)
        im_neg = np.mean(CAMSCI_STREAM.grab_many(NCAMSCI), axis=0)

        # im_pos_ni = normalize_coro_im(im_pos, im_params, ref_psf_params, dark_im=dark_im)
        # im_neg_ni = normalize_coro_im(im_neg, im_params, ref_psf_params, dark_im=dark_im)
        # diff_im_ni = im_pos_ni - im_neg_ni

        diff_im = im_pos - im_neg
        diff_im_ni = coro_utils.normalize_coro_im(diff_im, im_params, ref_psf_params, dark_im=0.0)

        all_ims.append([im_pos, im_neg])
        probed_responses.append( diff_im_ni / (2 * probe_amplitude) ) 

        if plot:
            utils.imshow(
                [probe_modes[i], diff_im_ni], 
                titles=[f'Probe Command {i+1}', 'Normalized Response', ],
                cmaps=['viridis', 'magma'], 
            )

    all_ims = np.array(all_ims)
    probed_responses = np.array(probed_responses)
    DM_STREAM.write( current_command*1e6 )
    
    return probed_responses
    
def calibrate(
        CAMSCI_STREAM, 
        NCAMSCI,
        DM_STREAM, 
        im_params,
        ref_psf_params,
        control_mask, 
        probe_amplitude, 
        probe_modes, 
        calibration_amplitude, 
        calibration_modes,
        delay=0.01,
        dark_im=0.0,
        scale_factors=None, 
        plot_responses=False, 
    ):
    print('Calibrating iEFC...')

    Nact = probe_modes.shape[1]
    Nprobes = probe_modes.shape[0]
    Nmodes = calibration_modes.shape[0]
    Ncamsci = CAMSCI_STREAM.shape[0]

    current_command = DM_STREAM.grab_latest()

    response_matrix = []
    calib_amps = []
    response_cube = []
    
    # Loop through all modes that you want to control
    start = time.time()
    for i, calibration_mode in enumerate(calibration_modes):
        response = 0
        for s in [-1, 1]: # We need a + and - probe to estimate the jacobian
            dm_mode = calibration_mode.reshape(Nact, Nact)
            amp = calibration_amplitude * scale_factors[i] if scale_factors is not None else calibration_amplitude
            calib_mode = ensure_np_array(amp * dm_mode)

            DM_STREAM.write( (current_command + s * calib_mode*1e6))
            time.sleep(delay)
            # Compute reponse with difference images of probes
            probed_diffs = take_measurement(
                CAMSCI_STREAM, 
                NCAMSCI,
                DM_STREAM, 
                im_params,
                ref_psf_params,
                # dark_im,
                probe_modes,
                probe_amplitude, 
                delay=delay,
            )
            calib_amps.append(amp)
            response += s * probed_diffs.reshape(Nprobes, Ncamsci**2) / (2 * amp)
            
            # DM_STREAM.write( (current_command - s * calib_mode) * 1e6) # Remove the mode from the DMs
        
        print(f"\tCalibrated mode {i+1:d}/{calibration_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")
        
        DM_STREAM.write( current_command )
        if probe_modes.shape[0]==2:
            response_matrix.append( np.concatenate([response[0, control_mask.ravel()],
                                                    response[1, control_mask.ravel()]]) )
        elif probe_modes.shape[0]==3: # if 3 probes are being used
            response_matrix.append( np.concatenate([response[0, control_mask.ravel()], 
                                                    response[1, control_mask.ravel()],
                                                    response[2, control_mask.ravel()]]) )
        
        response_cube.append(response)
    print('\nCalibration complete.')

    response_matrix = np.array(response_matrix).T # this is the response matrix to be inverted
    response_cube = np.array(response_cube)
    
    if plot_responses:
        dm_response_map = np.sqrt(np.mean(np.square(response_matrix.dot(calibration_modes.reshape(Nmodes, -1))), axis=0))
        dm_response_map = dm_response_map.reshape(Nact,Nact) / np.max(dm_response_map)

        fp_response_map = np.sqrt( np.mean( np.abs(response_cube), axis=(0,1))).reshape(Ncamsci, Ncamsci)
        fp_response_map = fp_response_map / np.max(fp_response_map)
        utils.imshow(
            [dm_response_map, fp_response_map], 
            titles=['DM Response Map', 'Focal Plane Response Map'],
            norms=[LogNorm(1e-2), None]
        )
            
    return response_matrix, response_cube
    
def run(iefc_data,
        CAMSCI_STREAM,
        NCAMSCI,
        DM_STREAM,
        im_params,
        ref_psf_params, 
        dark_im, 
        control_matrix,
        probe_amplitude, 
        probe_modes, 
        calib_modes,
        control_mask,
        delay=0.01,
        num_iterations=3,
        gain=0.75, 
        leakage=0.0,
        plot_current=True,
        plot_all=False,
        vmin=1e-9,
    ):
    
    start = time.time()
    starting_itr = len(iefc_data['images'])

    Nact = probe_modes.shape[1]
    Nmodes = calib_modes.shape[0]
    modal_matrix = calib_modes.reshape(Nmodes, -1).T

    total_command = copy.copy(iefc_data['commands'][-1]) if len(iefc_data['commands'])>0 else np.zeros((Nact,Nact))

    for i in range(num_iterations):
        print(f"Running iteration {i+starting_itr} / {num_iterations+starting_itr-1}")
        diff_ims = take_measurement(
            CAMSCI_STREAM, 
            NCAMSCI,
            DM_STREAM, 
            im_params,
            ref_psf_params,
            # dark_im,
            probe_modes,
            probe_amplitude, 
            delay=delay,
        )
        measurement_vector = diff_ims[:, control_mask].ravel()

        modal_coeff = -control_matrix.dot(measurement_vector)
        del_command = gain * modal_matrix.dot(modal_coeff).reshape(Nact, Nact)
        total_command = (1.0 - leakage) * total_command + del_command
        
        DM_STREAM.write( total_command * 1e6 )
        time.sleep(delay)

        print(f"Measuring dark hole state ...")
        coro_im = np.mean(CAMSCI_STREAM.grab_many(NCAMSCI), axis=0)
        coro_im_ni = coro_utils.normalize_coro_im(coro_im, im_params, ref_psf_params, dark_im)
        contrast = coro_utils.compute_contrast(coro_im_ni, control_mask)

        iefc_data['images'].append(copy.copy(coro_im_ni))
        iefc_data['contrasts'].append(copy.copy(contrast))
        iefc_data['commands'].append(copy.copy(total_command))
        iefc_data['del_commands'].append(copy.copy(del_command))
    
        if plot_current: 
            if not plot_all: clear_output(wait=True)
            utils.imshow(
                [del_command, total_command, coro_im_ni], 
                titles=[f'Iteration {starting_itr + i:d}: $\delta$DM', 
                        'Total DM Command', 
                        f'Normalized Image\nMean Contrast = {contrast:.3e}'],
                cmaps=['viridis', 'viridis', 'magma'],
                pxscls=[None, None, None],
                norms=[CenteredNorm(), None, LogNorm(vmin=vmin)],
            )

    print(f'Completed {num_iterations:d} iterations in {time.time()-start:.3f}s.')
    return iefc_data

def compute_hadamard_scale_factors(had_modes, scale_exp=1/6, scale_thresh=4, iwa=2.5, owa=13, oversamp=4, plot=False):
    Nact = had_modes.shape[1]

    ft_modes = []
    for i in range(had_modes.shape[0]):
        had_mode = had_modes[i]
        ft_modes.append(xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(utils.pad_or_crop(had_mode, Nact*oversamp)))))
    mode_freqs = xp.abs(xp.array(ft_modes))

    mode_freq_mask_pxscl = 1/oversamp
    x = (xp.linspace(-Nact*oversamp//2, Nact*oversamp//2-1, Nact*oversamp) + 1/2)*mode_freq_mask_pxscl
    x,y = xp.meshgrid(x,x)
    r = xp.sqrt(x**2+y**2)
    mode_freq_mask = (r>iwa)*(r<owa)
    if plot: utils.imshow([mode_freq_mask], pxscls=[1/oversamp])

    sum_vals = []
    max_vals = []
    for i in range(had_modes.shape[0]):
        sum_vals.append(xp.sum(mode_freqs[i, mode_freq_mask]))
        max_vals.append(xp.max(mode_freqs[i, mode_freq_mask]**2))

    biggest_sum = xp.max(xp.array(sum_vals))
    biggest_max = xp.max(xp.array(max_vals))

    scale_factors = []
    for i in range(had_modes.shape[0]):
        scale_factors.append((biggest_max/max_vals[i])**scale_exp)
        # scale_factors.append((biggest_sum/sum_vals[i])**(1/2))
    scale_factors = ensure_np_array(xp.array(scale_factors))

    scale_factors[scale_factors>scale_thresh] = scale_thresh
    if plot: 
        plt.plot(scale_factors)
        plt.show()

    return scale_factors







