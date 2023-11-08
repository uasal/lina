from .math_module import xp
from . import utils, scc
from . import imshows

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

def build_jacobian(sysi, 
                   wavelengths,
                   calibration_modes, calibration_amp,
                   control_mask, 
                   plot=False,
                  ):
    start = time.time()
    
    amps = np.linspace(-calibration_amp, calibration_amp, 2) # for generating a negative and positive actuator poke
    
    Nwaves = len(wavelengths)
    Nmodes = calibration_modes.shape[0]
    Nmask = int(control_mask.sum())
    
    responses_all = xp.zeros((2*Nwaves*Nmask, Nmodes))
    for i in range(Nwaves):
        print(f'Calculating Jacobian for wavelength {wavelengths[i]:.2e}: ')
        sysi.wavelength = wavelengths[i]
        responses = xp.zeros((2*Nmask, Nmodes))
        for i,mode in enumerate(calibration_modes):
            response = 0
            for amp in amps:
                mode = mode.reshape(sysi.Nact,sysi.Nact)

                sysi.add_dm(amp*mode)
                wavefront = sysi.calc_psf()
                response += amp * wavefront.flatten() / (2*np.var(amps))
                sysi.add_dm(-amp*mode)

            responses[::2,i] = response[control_mask.ravel()].real
            responses[1::2,i] = response[control_mask.ravel()].imag

            print('\tCalculated response for mode {:d}/{:d}. Elapsed time={:.3f} sec.'.format(i+1, Nmodes, time.time()-start), end='')
            print("\r", end="")
        responses_all[i:(i+1)*Nwaves] = responses
    
    print()
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    # if plot:
    #     total_response = responses[::2] + 1j*responses[1::2]
    #     dm_response = total_response.dot(xp.array(calibration_modes))
    #     dm_response = xp.sqrt(xp.mean(xp.abs(dm_response)**2, axis=0)).reshape(sysi.Nact, sysi.Nact)
    #     imshows.imshow1(dm_response, lognorm=True, vmin=dm_response.max()*1e-2)

    return responses


def run_efc_perfect(sysi, 
                    wavelengths,
                    jac, 
                    calibration_modes,
                    control_matrix,
                    control_mask, 
                    pwp_fun=None,
                    pwp_params=None,
                    loop_gain=0.5, 
                    leakage=0.0,
                    iterations=5, 
                    plot_all=False, 
                    plot_current=True,
                    plot_sms=False,
                    plot_radial_contrast=False,
                    old_images=None,
                    old_fields=None,
                    old_commands=None,
                    ):
    # This function is only for running EFC simulations
    print('Beginning closed-loop EFC.')    
    start = time.time()
    
    U, s, V = xp.linalg.svd(jac, full_matrices=False)
    alpha2 = xp.max( xp.diag( xp.real( jac.conj().T @ jac ) ) )
    print('Max singular value squared:\t', s.max()**2)
    print('alpha^2:\t\t\t', alpha2) 
    
    calibration_modes = xp.array(calibration_modes)

    Nact = sysi.Nact
    Nmask = int(control_mask.sum())
    
    # The metrics
    metric_images = []
    fields = []
    dm_commands = []

    dm_ref = sysi.get_dm()
    command = 0.0
    dm_command = 0.0

    if old_images is None:
        starting_iteration = 0
    else:
        starting_iteration = len(old_images) - 1

    for i in range(iterations):
        print(f'\tRunning iteration {i+1+starting_iteration}/{iterations+starting_iteration}.')
        
        if pwp_fun is not None and pwp_params is not None:
            print('Using PWP to estimate electric field')
            electric_field = pwp_fun(sysi, control_mask, **pwp_params)
        else:
            print('Using model to compute electric field at each wavelength')
            for i in range(len(wavelengths)):
                sysi.wavelength = wavelengths[i]
                electric_field = sysi.calc_psf() # no PWP, just use model
        
        efield_ri = xp.zeros(2*Nmask)

        efield_ri[::2] = electric_field[control_mask].real
        efield_ri[1::2] = electric_field[control_mask].imag

        modal_coefficients = -control_matrix.dot(efield_ri)
        command = (1.0-leakage)*command + loop_gain*modal_coefficients
        
        # Reconstruct the full phase from the Fourier modes
        act_commands = calibration_modes.T.dot(command)
        dm_command = act_commands.reshape(sysi.Nact,sysi.Nact)

        # Set the current DM state
        sysi.set_dm(dm_ref + dm_command)
        
        # Take an image to estimate the metrics
        # electric_field = sysi.calc_psf()
        # image = xp.abs(electric_field)**2
        image = sysi.snap()

        # efields.append([copy.copy(electric_field)])
        metric_images.append(copy.copy(image))
        fields.append(copy.copy(electric_field))
        dm_commands.append(sysi.get_dm())
        
        mean_ni = xp.mean(image.ravel()[control_mask.ravel()])

        if plot_current or plot_all:

            imshows.imshow2(dm_commands[i], image, 
                            'DM', f'Image: Iteration {i+starting_iteration+1}\nMean NI: {mean_ni:.3e}',
                            cmap1='viridis',
                            lognorm2=True, vmin2=1e-11, pxscl2=sysi.psf_pixelscale_lamD, xlabel2='$\lambda/D$')

            if plot_radial_contrast:
                utils.plot_radial_contrast(image, control_mask, sysi.psf_pixelscale_lamD, nbins=100)
            
            if not plot_all: clear_output(wait=True)

    metric_images = xp.array(metric_images)
    fields = xp.array(fields)
    dm_commands = xp.array(dm_commands)
    
    if old_images is not None:
        metric_images = xp.concatenate([old_images, metric_images], axis=0)
    if old_fields is not None:
        fields = xp.concatenate([old_fields, fields], axis=0)
    if old_commands is not None: 
        dm_commands = xp.concatenate([old_commands, dm_commands], axis=0)
                
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return metric_images, fields, dm_commands





