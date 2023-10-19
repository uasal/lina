from .math_module import xp, _scipy, ensure_np_array
from . import utils
from . import imshows

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output


# def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
def take_measurement(sysi, probe_cube, probe_amplitude, return_all=False, pca_modes=None, plot=False):
    """_summary_

    Parameters
    ----------
    sysi : object
        The object of a system interface with methods for DM control and image 
    probe_cube : xp.ndarray
        3D array of probes to measure difference images of, shape of (Nprobes, Nact, Nact)
    probe_amplitude : _type_
        _description_
    return_all : bool, optional
        _description_, by default False
    pca_modes : _type_, optional
        _description_, by default None
    plot : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """

    differential_operator = []
    for i in range(len(probe_cube)):
        vec = [0]*2*len(probe_cube)
        vec[2*i] = -1
        vec[2*i+1] = 1
        differential_operator.append(vec)
    differential_operator = xp.array(differential_operator) / (2 * probe_amplitude)
    
    amps = np.linspace(-probe_amplitude, probe_amplitude, 2)
    images = []
    for probe in probe_cube: 
        for amp in amps:
            sysi.add_dm(amp*probe)
            image = sysi.snap()
            images.append(image.flatten())
            sysi.add_dm(-amp*probe)
    images = xp.array(images)
    
    differential_images = differential_operator.dot(images)
    
    if plot:
        for i, diff_im in enumerate(differential_images):
            imshows.imshow2(probe_cube[i], diff_im.reshape(sysi.npsf, sysi.npsf), 
                            f'Probe Command {i+1}', 'Difference Image', pxscl2=sysi.psf_pixelscale_lamD,
                            cmap1='viridis')
            
    if pca_modes is not None:
        differential_images = differential_images - (pca_modes.T.dot( pca_modes.dot(differential_images.T) )).T
        
    if return_all:
        return differential_images, images
    else:
        return differential_images

def calibrate(sysi, 
              control_mask, 
              probe_amplitude, probe_modes,
              calibration_amplitude, calibration_modes, 
              start_mode=0,
              return_all=False, 
             plot_sum=False):
    print('Calibrating iEFC...')
    Nmodes = calibration_modes.shape[0]
    
    response_cube = []
    if return_all:
        response_matrix = []
    # Loop through all modes that you want to control
    start = time.time()
    for ci, calibration_mode in enumerate(calibration_modes[start_mode::]):
        response = 0
        for s in [-1, 1]:
            # Set the DM to the correct state
            sysi.add_dm(s * calibration_amplitude * calibration_mode.reshape(sysi.Nact, sysi.Nact))
            differential_images = take_measurement(sysi, probe_modes, probe_amplitude, return_all=False)
            response += s * differential_images / (2 * calibration_amplitude)
            sysi.add_dm(-s * calibration_amplitude * calibration_mode.reshape(sysi.Nact, sysi.Nact))

        print(f'\tCalibrated mode {ci+1+start_mode} / {Nmodes} in {time.time()-start:.2f}s', end='')
        print('\r', end='')
        
        measured_response = []
        for i in range(probe_modes.shape[0]):
            measured_response.append(response[i, control_mask.ravel()])
        measured_response = xp.array(measured_response)
#         print(measured_response.shape)
        response_matrix.append(xp.concatenate(measured_response)) # masked response for each probe mode 
        if return_all:
            response_cube.append(response)
    print()
    print('Calibration complete.')
    
    response_matrix = xp.array(response_matrix).T
    if return_all:
        response_cube = xp.array(response_cube)
    
    if plot_sum:
        dm_rss = xp.sqrt(xp.sum(abs(response_matrix.dot(xp.array(calibration_modes)))**2, axis=0)).reshape(sysi.Nact,sysi.Nact)
        imshows.imshow1(dm_rss, 'DM RSS Response')
        if return_all:
            fp_rss = xp.sqrt(xp.sum(abs(response_cube)**2, axis=(0,1))).reshape(sysi.npsf,sysi.npsf)
            imshows.imshow1(fp_rss, 'Focal Plane RSS Response', lognorm=True)
            
    if return_all:
        return response_matrix, response_cube
    else:
        return response_matrix

def single_iteration(sysi, probe_cube, probe_amplitude, control_matrix, control_mask):
    # Take a measurement
    differential_images = take_measurement(sysi, probe_cube, probe_amplitude)
    
    # Choose which pixels we want to control
    measurement_vector = differential_images[:, control_mask.ravel()].ravel()
    print(measurement_vector.shape)
    print(control_matrix.shape)
    # Calculate the control signal in modal coefficients
    reconstructed_coefficients = control_matrix.dot( measurement_vector )
    
    return reconstructed_coefficients
    
def run(sysi,  
        control_matrix, 
        probe_modes, probe_amplitude,
        calibration_modes, 
        control_mask,
        num_iterations=10, 
        loop_gain=0.5, 
        leakage=0.0,
        plot_current=True,
        plot_all=False,
        plot_radial_contrast=True,
        old_images=None,
        old_dm_commands=None,):
    print('Running iEFC...')
    start = time.time()
    
    dm_commands = np.zeros((num_iterations, sysi.Nact, sysi.Nact), dtype=np.float64)
    metric_images = xp.zeros((num_iterations, sysi.npsf, sysi.npsf), dtype=xp.float64)
    
    dm_ref = sysi.get_dm()
    modal_coeff = 0.0
    for i in range(num_iterations):
        print(f"\tClosed-loop iteration {i+1} / {num_iterations}")
        
        delta_coefficients = single_iteration(sysi, probe_modes, probe_amplitude, control_matrix, control_mask)
        modal_coeff = (1.0-leakage)*modal_coeff + loop_gain*delta_coefficients
        
        # Reconstruct the full phase from the Fourier modes
        dm_command = -calibration_modes.T.dot(ensure_np_array(modal_coeff)).reshape(sysi.Nact,sysi.Nact)
        sysi.set_dm(dm_ref + dm_command)
        
        # Take an image to estimate the metrics
        image = sysi.snap()
        
        metric_images[i] = image
        dm_commands[i] = sysi.get_dm()
        
        mean_ni = xp.mean(image.ravel()[control_mask.ravel()])
        print(f'\tMean NI of this iteration: {mean_ni:.3e}')
        
        if plot_current: 
            if not plot_all: clear_output(wait=True)
            imshows.imshow3(dm_commands[i], image, image*control_mask,
                            'DM Command', f'Image: Iteration {i}',
                            pxscl2=sysi.psf_pixelscale_lamD, pxscl3=sysi.psf_pixelscale_lamD, 
                            lognorm2=True, lognorm3=True
#                             vmin2=1e-11,
                           )
            if plot_radial_contrast:
                utils.plot_radial_contrast(image, control_mask, sysi.psf_pixelscale_lamD, nbins=100)
                
    if old_images is not None:
        metric_images = xp.concatenate([old_images, metric_images], axis=0)
    if old_dm_commands is not None:
        dm_commands = xp.concatenate([old_dm_commands, xp.array(dm_commands)], axis=0)
        
    print('iEFC loop completed in {:.3f}s.'.format(time.time()-start))
    return metric_images, dm_commands


