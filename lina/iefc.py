from .math_module import xp, xcipy, ensure_np_array
from lina import utils, coro_utils

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize, CenteredNorm

def measure_probe_response(
        take_im_fun,
        take_im_params,
        set_dm_fun,
        set_dm_params,
        probe_modes,
        probe_amplitude, 
        base_command=None,
        normalize_diff_fun=None,
        normalize_diff_params=None,
        verbose=False,
        plot=False,
    ):
    """
    This will measure the difference images for a provided set of DM probes. 

    Args:
        take_im_fun (callable): 
            Function that returns the image of the coronagraph.  
        take_im_params (dict): 
            Dictionary of additional parameters needed for the take_im_fun method.
        set_dm_fun (callable): 
            Function that applies the DM command to the coronagraph. First argument of 
            this function must be the DM command that will be applied. 
        set_dm_params (dict): 
            Dictionary of additional parameters needed for the set_dm_fun method.
        probe_modes (ndarray):
            Cube of the DM probe modes that is of the shape Nprobes X Nact X Nact.
        probe_amplitude (float): 
            Amplitude to apply to the probes in units of meters.
        base_command (ndarray, optional): 
            Underlying command that the probes will be added to. Defaults to None.
        normalize_diff_fun (callable, optional): 
            Function that normalizes the difference images of the probes. If take_im_fun
            automatically returns normalized intensity images, this is not needed. Defaults to None.
        normalize_diff_params (dict, optional): 
            Dictionary of additional parameters needed for the normalize_diff_fun method. Defaults to None.
        plot (bool, optional): 
            Plot the normalized difference images as they are measured. Defaults to False.

    Returns:
        xp.ndarray: Data cube of the probed difference images of the shape Nprobes X Ncam X Ncam
    """

    Nprobes = probe_modes.shape[0]
    Nact = probe_modes.shape[1]
    if base_command is None: base_command = xp.zeros((Nact, Nact))
    
    all_ims = []
    probed_responses = []
    for i in range(Nprobes):
        if verbose:
            print(f'\tMeasuring response of probe {i+1}/{Nprobes}.')
        probe = probe_amplitude * probe_modes[i]

        set_dm_fun(base_command + probe, **set_dm_params)
        im_pos = take_im_fun(**take_im_params)

        set_dm_fun(base_command - probe, **set_dm_params)
        im_neg = take_im_fun(**take_im_params)

        diff_im = im_pos - im_neg
        diff_im_ni = diff_im if normalize_diff_fun is None else normalize_diff_fun(diff_im, **normalize_diff_params)

        all_ims.append([im_pos, im_neg])
        probed_responses.append( diff_im_ni / (2 * probe_amplitude) ) 

        if plot:
            utils.imshow(
                [probe_modes[i], diff_im_ni], 
                titles=[f'Probe Command {i+1}', 'Normalized Response', ],
                cmaps=['viridis', 'magma'], 
            )

    all_ims = xp.array(all_ims)
    probed_responses = xp.array(probed_responses)
    set_dm_fun(base_command, **set_dm_params)
    
    return probed_responses
    
def calibrate(
        take_im_fun,
        take_im_params,
        set_dm_fun,
        set_dm_params,
        wfs_mask, 
        probe_modes, 
        probe_amplitude, 
        calibration_modes,
        calibration_amplitude,
        scale_factors=None,  
        initial_command=None,
        normalize_diff_fun=None,
        normalize_diff_params=None,
        plot_responses=False, 
    ):
    """
    This function will calibrate the coronagraph for a given set of calibration modes with the 
    given set of DM probes. 

    Args:
        take_im_fun (callable): 
            Function that returns the image of the coronagraph.  
        take_im_params (dict): 
            Dictionary of additional parameters needed for the take_im_fun method.
        set_dm_fun (callable): 
            Function that applies the DM command to the coronagraph. First argument of 
            this function must be the DM command that will be applied. 
        set_dm_params (dict): 
            Dictionary of additional parameters needed for the set_dm_fun method.
        wfs_mask (ndarray):
            Binary mask defining the region in the focal plane to control.
        probe_modes (ndarray):
            Cube of the DM probe modes that is of the shape Nprobes X Nact X Nact.
        probe_amplitude (float): 
            Amplitude to apply to the probes in units of meters.
        calibration_modes (ndarray): 
            Cube of the DM calibration modes that is of the shape Nmodes X Nact X Nact.
        calibration_amplitude (float): 
            Amplitude to apply to the calibration modes in units of meters. 
        scale_factors (ndarray, optional): 
            Vector of scale factors that are applied to each respective calibration mode. 
            Allows for different modes to use different calibration amplitudes to prevent
            saturation on concentrated modes but good SNR on distributed modes. Defaults to None.
        initial_command (ndarray, optional): 
            Underlying command that the calibration modes will be added to. Defaults to None.
        normalize_diff_fun (callable, optional): 
            Function that normalizes the difference images of the probes. If take_im_fun
            automatically returns normalized intensity images, this is not needed. Defaults to None.
        normalize_diff_params (dict, optional): 
            Dictionary of additional parameters needed for the normalize_diff_fun method. Defaults to None.
        plot_responses (bool, optional): 
            Plots the response maps in DM space and in WFS space. Defaults to False.

    Returns:
        tuple: 
            Returns the response matrix for the region of interest specified by the wfs_mask and
            the response cube containing the complete response measurements for the entire focal plane. 

    """
    print('Calibrating iEFC...')

    Nact = probe_modes.shape[1]
    Nprobes = probe_modes.shape[0]
    Nmodes = calibration_modes.shape[0]
    Ncamsci = wfs_mask.shape[0]
    if initial_command is None: initial_command = xp.zeros((Nact, Nact))

    response_matrix = []
    calib_amps = []
    response_cube = []
    
    # Loop through all modes that you want to control
    start = time.time()
    for i, calibration_mode in enumerate(calibration_modes):
        response = 0
        for s in [1, -1]: # We need a + and - probe to estimate the jacobian
            dm_mode = calibration_mode.reshape(Nact, Nact)
            amp = calibration_amplitude * scale_factors[i] if scale_factors is not None else calibration_amplitude
            calib_mode = amp * dm_mode

            base_command = initial_command + s*calib_mode
            # Compute reponse with difference images of probes
            probed_diffs = measure_probe_response(
                take_im_fun,
                take_im_params,
                set_dm_fun,
                set_dm_params,
                probe_modes,
                probe_amplitude, 
                base_command=base_command,
                normalize_diff_fun=normalize_diff_fun,
                normalize_diff_params=normalize_diff_params,
            )
            calib_amps.append(amp)
            # response += s * probed_diffs.reshape(Nprobes, Ncamsci**2) / (2 * amp)
            response += s * probed_diffs / (2 * amp)
            # print(type(response))
            
        print(f"\tCalibrated mode {i+1:d}/{calibration_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")
        
        set_dm_fun(initial_command, **set_dm_params)
        raveled_response = response[:, wfs_mask].ravel()
        response_matrix.append(raveled_response)
        response_cube.append(response)
    print('\nCalibration complete.')

    response_matrix = xp.array(response_matrix).T # this is the response matrix to be inverted
    response_cube = xp.array(response_cube)
    
    if plot_responses:
        dm_response_map = xp.sqrt(xp.mean(xp.square(response_matrix.dot(calibration_modes.reshape(Nmodes, -1))), axis=0))
        dm_response_map = dm_response_map.reshape(Nact,Nact) / xp.max(dm_response_map)

        fp_response_map = xp.sqrt( xp.mean( xp.abs(response_cube), axis=(0,1))).reshape(Ncamsci, Ncamsci)
        fp_response_map = fp_response_map / xp.max(fp_response_map)
        utils.imshow(
            [dm_response_map, fp_response_map], 
            titles=['DM Response Map', 'Focal Plane Response Map'],
            norms=[LogNorm(1e-2), None]
        )

    return response_matrix, response_cube
    
def make_response_matrix(
        response_cube,
        wfs_mask,
    ):
    Nmodes = response_cube.shape[0]
    response_matrix = response_cube[:, :, wfs_mask].reshape(Nmodes, -1).T
    return response_matrix

def init_data(
        wfs_mask=None, 
        contrast0=None,
        ni_im0=None,
    ):
    iefc_data = {
        'raw_images':[],
        'ni_images':[],
        'contrasts':[],
        'commands':[],
        'del_commands':[],
        'reg_conds':[],
        'wfs_mask':wfs_mask,
        'ni_im0':ni_im0,
        'contrast0':contrast0,
    }
    return iefc_data

def run(iefc_data,
        take_im_fun,
        take_im_params,
        set_dm_fun,
        set_dm_params,
        # control_matrix,
        response_matrix,
        reg_cond,
        probe_modes,
        probe_amplitude, 
        calib_modes,
        wfs_mask,
        num_iterations=3,
        gain=1.0, 
        leakage=0.0,
        normalize_diff_fun=None,
        normalize_diff_params=None,
        normalize_metric_fun=None,
        normalize_metric_params=None,
        plot_current=True,
        plot_all=False,
        verbose=True, 
        plot_probe_responses=False,
        vmin=1e-10,
        vmax=1e-5,
    ):
    """_summary_

    Args:
        iefc_data (dict): 
            Dictionary of all the corresponding data for this particular iEFC run. Dictionary contains 
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
        response_matrix (ndarray): 
            Response matrix for the region of interest specified by the wfs_mask. 
        reg_cond (float):
            Regularization value to perform the pseudo-inverse of the response matrix. 
            Also known as the beta value since the beta regularization method is used here. 
        probe_modes (ndarray): 
            Cube of the DM probe modes used for the provided control matrix. 
        probe_amplitude (float): 
            Amplitude to apply to the DM probes for each iteration of iEFC. 
        calibration_modes (ndarray): 
            Cube of the DM calibration modes used for the provided control matrix.
        wfs_mask (ndarray):
            Binary mask defining the region in the focal plane to control.
        num_iterations (int, optional): 
            Number of iterati9ons to perform iEFC with these specific parameters. Defaults to 3.
        gain (float, optional): 
            Loop gain applied to each computed DM command. Defaults to 1.0.
        leakage (float, optional): 
            Leakage specifiy how much of the previous commands to remove. Defaults to 0.0.
        normalize_diff_fun (callable, optional): 
            Function that normalizes the difference images of the probes. If take_im_fun
            automatically returns normalized intensity images, this is not needed. Defaults to None.
        normalize_diff_params (dict, optional): 
            Dictionary of additional parameters needed for the normalize_diff_fun method. Defaults to None.
        normalize_metric_fun (callable, optional): 
            Function that normalizes the metric image used to evaluate current contrast. If take_im_fun
            automatically returns normalized intensity images, this is not needed. Defaults to None.
        normalize_metric_params (dict, optional): 
            Dictionary of additional parameters needed for the normalize_metric_fun method. Defaults to None.
        plot_current (bool, optional): 
            Plots the results of the current iteration. Defaults to True.
        plot_all (bool, optional): 
            Plots the results of all iterations performed during this round of iEFC. Defaults to False.
        vmin (float, optional): 
            Minimum contrast value to display on the plots. Defaults to 1e-9.

    Returns:
        iefc_data (dict): 
            Dictionary of iEFC data appended with the results of the new iterations performed. 
    """
    
    start = time.time()

    Nact = probe_modes.shape[1]
    Nmodes = calib_modes.shape[0]
    modal_matrix = calib_modes.reshape(Nmodes, -1).T

    starting_itr = len(iefc_data['commands']) + 1
    total_command = copy.copy(iefc_data['commands'][-1]) if len(iefc_data['commands'])>0 else xp.zeros((Nact,Nact))
    
    control_matrix = utils.beta_reg(response_matrix, reg_cond)

    for i in range(num_iterations):
        print(f"Running iteration {i+starting_itr} / {num_iterations+starting_itr-1}")
        diff_ims = measure_probe_response(
            take_im_fun,
            take_im_params,
            set_dm_fun,
            set_dm_params,
            probe_modes,
            probe_amplitude, 
            base_command=total_command,
            normalize_diff_fun=normalize_diff_fun,
            normalize_diff_params=normalize_diff_params,
            verbose=verbose,
            plot=plot_probe_responses,
        )
        measurement_vector = diff_ims[:, wfs_mask].ravel()

        modal_coeff = -control_matrix.dot(measurement_vector)
        del_command = gain * modal_matrix.dot(modal_coeff).reshape(Nact, Nact)
        total_command = (1.0 - leakage) * total_command + del_command
        
        set_dm_fun(total_command, **set_dm_params)

        print(f"Measuring dark hole state ...")
        metric_im = take_im_fun(**take_im_params)
        metric_im_ni = metric_im if normalize_metric_fun is None else normalize_metric_fun(metric_im, **normalize_metric_params)
        contrast = coro_utils.compute_contrast(metric_im_ni, wfs_mask)

        iefc_data['raw_images'].append(copy.copy(metric_im))
        iefc_data['ni_images'].append(copy.copy(metric_im_ni))
        iefc_data['contrasts'].append(copy.copy(contrast))
        iefc_data['commands'].append(copy.copy(total_command))
        iefc_data['del_commands'].append(copy.copy(del_command))
        iefc_data['reg_conds'].append(copy.copy(reg_cond))
    
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



