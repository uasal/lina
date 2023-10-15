from math_module import xp, _scipy, ensure_np_array
import utils
import imshows

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

from pathlib import Path
# iefc_data_dir = Path('/groups/douglase/kians-data-files/roman-cgi-iefc-data')
iefc_data_dir = Path('/home/kianmilani/Projects/roman-cgi-iefc-data')

# def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
def take_measurement(sysi, probe_cube, probe_amplitude, DM=1, return_all=False, pca_modes=None, plot=False):
    
    differential_operator = []
    for i in range(len(probe_cube)):
        vec = [0]*2*len(probe_cube)
        vec[2*i] = -1
        vec[2*i+1] = 1
        differential_operator.append(vec)
    differential_operator = xp.array(differential_operator) / (2 * probe_amplitude)
#     print(differential_operator)
    
    amps = np.linspace(-probe_amplitude, probe_amplitude, 2)
    images = []
    for probe in probe_cube: 
        for amp in amps:
            if DM==1:
                sysi.add_dm1(amp*probe)
                image = sysi.snap()
                images.append(image.flatten())
                sysi.add_dm1(-amp*probe)
            elif DM==2:
                sysi.add_dm2(amp*probe)
                image = sysi.snap()
                images.append(image.flatten())
                sysi.add_dm2(-amp*probe)
            
    images = xp.array(images)
    
    differential_images = differential_operator.dot(images)
    
    if pca_modes is not None:
        differential_images = differential_images - (pca_modes.T.dot( pca_modes.dot(differential_images.T) )).T
    
    if plot:
        for i, diff_im in enumerate(differential_images):
            imshows.imshow2(probe_cube[i], diff_im.reshape(sysi.npsf, sysi.npsf), 
                            f'Probe Command {i+1}', 'Difference Image', pxscl2=sysi.psf_pixelscale_lamD,
                            cmap1='viridis')
    
    if return_all:
        return differential_images, images
    else:
        return differential_images
    
def calibrate(sysi, 
              control_mask, 
              probe_amplitude, probe_modes, 
              calibration_amplitude, calibration_modes, 
              return_all=False,
              plot_responses=True):
    print('Calibrating iEFC...')
    
    response_matrix = []
    if return_all: # be ready to store the full focal plane responses (difference images)
        response_cube = []
    
    # Loop through all modes that you want to control
    start = time.time()
    for ci, calibration_mode in enumerate(calibration_modes):
        response = 0
        for s in [-1, 1]: # We need a + and - probe to estimate the jacobian
            # reshape calibration mode into the DM1 and DM2 components
            dm1_mode = calibration_mode[:sysi.Nact**2].reshape(sysi.Nact, sysi.Nact)
            dm2_mode = calibration_mode[sysi.Nact**2:].reshape(sysi.Nact, sysi.Nact)
            
            # Add the mode to the DMs
            sysi.add_dm1(s * calibration_amplitude * dm1_mode)
            sysi.add_dm2(s * calibration_amplitude * dm2_mode)
            
            # Compute reponse with difference images of probes
            diff_ims = take_measurement(sysi, probe_modes, probe_amplitude, DM=1)
            response += s * diff_ims / (2 * calibration_amplitude)
            
            # Remove the mode form the DMs
            sysi.add_dm1(-s * calibration_amplitude * dm1_mode) # remove the mode
            sysi.add_dm2(-s * calibration_amplitude * dm2_mode) 
        
        print("\tCalibrated mode {:d}/{:d} in {:.3f}s".format(ci+1, calibration_modes.shape[0], time.time()-start), end='')
        print("\r", end="")
        
        if probe_modes.shape[0]==2:
            response_matrix.append( xp.concatenate([response[0, control_mask.ravel()],
                                                    response[1, control_mask.ravel()]]) )
        elif probe_modes.shape[0]==3: # if 3 probes are being used
            response_matrix.append( xp.concatenate([response[0, control_mask.ravel()], 
                                                    response[1, control_mask.ravel()],
                                                    response[2, control_mask.ravel()]]) )
        
        if return_all: 
            response_cube.append(response)
            
    response_matrix = xp.array(response_matrix).T # this is the response matrix to be inverted
    
    if return_all:
        response_cube = xp.array(response_cube)
    print()
    print('Calibration complete.')
    
    if plot_responses:
        dm_rms = xp.sqrt(xp.mean((response_matrix.dot(xp.array(calibration_modes)))**2, axis=0))
        dm1_rms = dm_rms[:sysi.Nact**2].reshape(sysi.Nact, sysi.Nact)
        dm2_rms = dm_rms[sysi.Nact**2:].reshape(sysi.Nact, sysi.Nact)
        imshows.imshow2(dm1_rms, dm2_rms, 
                        'DM1 RMS Actuator Responses', 'DM2 RMS Actuator Responses')
        if return_all:
            fp_rms = xp.sqrt(xp.mean(abs(response_cube)**2, axis=(0,1))).reshape(sysi.npsf,sysi.npsf)
            imshows.imshow1(fp_rms, 'Focal Plane Pixels RMS Response', lognorm=True)
            
    if return_all:
        return response_matrix, xp.array(response_cube)
    else:
        return response_matrix
    

def single_iteration(sysi, probe_cube, probe_amplitude, control_matrix, control_mask):
    # Take a measurement
    differential_images = take_measurement(sysi, probe_cube, probe_amplitude)
    
    # Choose which pixels we want to control
    measurement_vector = differential_images[:, control_mask.ravel()].ravel()

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
        use_fourier_filter=False,
        plot_current=True,
        plot_all=False,
        plot_radial_contrast=True,
        old_images=None,
        old_dm1_commands=None,
        old_dm2_commands=None,
       ):
    
    print('Running iEFC...')
    start = time.time()
    
    Nc = calibration_modes.shape[0]
    
    # The metric
    metric_images = []
    dm1_commands = []
    dm2_commands = []
    
    dm1_ref = sysi.get_dm1()
    dm2_ref = sysi.get_dm2()
    command = 0.0
    dm1_command = 0.0
    dm2_command = 0.0
    
    if old_images is None:
        starting_iteration = 0
    else:
        starting_iteration = len(old_images) - 1
        
    for i in range(num_iterations):
        print(f"\tClosed-loop iteration {i+1+starting_iteration} / {num_iterations+starting_iteration}")
        
        # delta_coefficients = single_iteration(sysi, probe_modes, probe_amplitude, control_matrix, control_mask)
        # Take a measurement
        differential_images = take_measurement(sysi, probe_modes, probe_amplitude)
        measurement_vector = differential_images[:, control_mask.ravel()].ravel()
        modal_coefficients = -control_matrix.dot( measurement_vector )

        command = (1.0-leakage)*command + loop_gain*modal_coefficients
        
        act_commands = calibration_modes.T.dot(utils.ensure_np_array(command))
        dm1_command = act_commands[:sysi.Nact**2].reshape(sysi.Nact,sysi.Nact)
        dm2_command = act_commands[sysi.Nact**2:].reshape(sysi.Nact,sysi.Nact)
        
        # Set the current DM state
        sysi.set_dm1(dm1_ref + dm1_command)
        sysi.set_dm2(dm2_ref + dm2_command)
        
        # Take an image to estimate the metrics
        image = sysi.snap()
        
        metric_images.append(copy.copy(image))
        dm1_commands.append(sysi.get_dm1())
        dm2_commands.append(sysi.get_dm2())
        
        mean_ni = xp.mean(image.ravel()[control_mask.ravel()])
        
        if plot_current: 
            if not plot_all: clear_output(wait=True)
            imshows.imshow3(dm1_commands[i], dm2_commands[i], image, 
                               'DM1', 'DM2', f'Image: Iteration {i+starting_iteration+1}\nMean NI: {mean_ni:.3e}',
                            cmap1='viridis', cmap2='viridis',
                               lognorm3=True, vmin3=1e-11, pxscl3=sysi.psf_pixelscale_lamD, xlabel3='$\lambda/D$')
            
            if plot_radial_contrast:
                utils.plot_radial_contrast(image, control_mask, sysi.psf_pixelscale_lamD, nbins=50,
#                                            ylims=[1e-10, 1e-4],
                                          )
                
    metric_images = xp.array(metric_images)
    dm1_commands = xp.array(dm1_commands)
    dm2_commands = xp.array(dm2_commands)
    
    if old_images is not None:
        metric_images = xp.concatenate([old_images, metric_images], axis=0)
    if old_dm1_commands is not None: 
        dm1_commands = xp.concatenate([old_dm1_commands, dm1_commands], axis=0)
    if old_dm2_commands is not None:
        dm2_commands = xp.concatenate([old_dm2_commands, dm2_commands], axis=0)
        
    print('Closed loop for given control matrix completed in {:.3f}s.'.format(time.time()-start))
    return metric_images, dm1_commands, dm2_commands






