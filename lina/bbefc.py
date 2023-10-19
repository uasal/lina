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
                    jac, 
                    calibration_modes,
                    control_matrix,
                    control_mask, 
                    Imax_unocc=1,
                    loop_gain=0.5, 
                    leakage=0.0,
                    iterations=5, 
                    plot_all=False, 
                    plot_current=True,
                    plot_sms=False,
                    plot_radial_contrast=False,
                    old_images=None,
                    old_dm_commands=None,
                    ):
    # This function is only for running EFC simulations
    print('Beginning closed-loop EFC simulation.')    
    commands = np.zeros((iterations, sysi.Nact, sysi.Nact), dtype=np.float64)
    efields = xp.zeros((iterations, sysi.npsf, sysi.npsf), dtype=xp.complex128)
    images = xp.zeros((iterations, sysi.npsf, sysi.npsf), dtype=xp.float64)
    start = time.time()
    
    U, s, V = xp.linalg.svd(jac, full_matrices=False)
    alpha2 = xp.max( xp.diag( xp.real( jac.conj().T @ jac ) ) )
    print('Max singular value squared:\t', s.max()**2)
    print('alpha^2:\t\t\t', alpha2) 
    
    # calibration_modes = xp.array(calibration_modes)

    Nact = sysi.Nact
    Nmask = int(control_mask.sum())
    
    # The metric
    # efields = []
    metric_images = []
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
        
        electric_field = sysi.calc_psf() # no PWP, just use model
        efield_ri = xp.zeros(2*Nmask)

        efield_ri[::2] = electric_field[control_mask].real
        efield_ri[1::2] = electric_field[control_mask].imag

        modal_coefficients = -control_matrix.dot(efield_ri)
        command = (1.0-leakage)*command + loop_gain*modal_coefficients
        
        # Reconstruct the full phase from the Fourier modes
        act_commands = calibration_modes.T.dot(utils.ensure_np_array(command))
        dm_command = act_commands.reshape(sysi.Nact,sysi.Nact)

        # Set the current DM state
        sysi.set_dm(dm_ref + dm_command)
        
        # Take an image to estimate the metrics
        # electric_field = sysi.calc_psf()
        # image = xp.abs(electric_field)**2
        image = sysi.snap()

        # efields.append([copy.copy(electric_field)])
        metric_images.append(copy.copy(image))
        dm_commands.append(sysi.get_dm())
        
        mean_ni = xp.mean(image.ravel()[control_mask.ravel()])
        # print(f'\tMean NI of this iteration: {mean_ni:.3e}')

        if plot_current or plot_all:

            imshows.imshow2(dm_commands[i], image, 
                               'DM', f'Image: Iteration {i+starting_iteration+1}\nMean NI: {mean_ni:.3e}',
                            cmap1='viridis',
                               lognorm2=True, vmin2=1e-11, pxscl2=sysi.psf_pixelscale_lamD, xlabel2='$\lambda/D$')

            if plot_sms:
                sms_fig = sms(U, s, alpha2, efield_ri, Nmask, Imax_unocc, i)

            if plot_radial_contrast:
                utils.plot_radial_contrast(image, control_mask, sysi.psf_pixelscale_lamD, nbins=100)
            
            if not plot_all: clear_output(wait=True)

    metric_images = xp.array(metric_images)
    dm_commands = xp.array(dm_commands)
    
    if old_images is not None:
        metric_images = xp.concatenate([old_images, metric_images], axis=0)
    if old_dm_commands is not None: 
        dm_commands = xp.concatenate([old_dm_commands, dm_commands], axis=0)
                
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return metric_images, dm_commands

# def build_jacobian(sysi, wavelengths, epsilon, dark_mask, display=False):
#     start = time.time()
#     print('Building Jacobian.')
    
#     responses = []
#     amps = np.linspace(-epsilon, epsilon, 2) # for generating a negative and positive actuator poke
    
#     dm_mask = sysi.dm_mask.flatten()
    
#     num_modes = sysi.Nact**2
#     modes = np.eye(num_modes) # each column in this matrix represents a vectorized DM shape where one actuator has been poked
    
#     for wavelength in wavelengths:
#         sysi.wavelength = wavelength
#         print('Calculating sensitivity for wavelength {:.3e}'.format(wavelength))
        
#         for i, mode in enumerate(modes):
#             if dm_mask[i]==1:
#                 response = 0
#                 for amp in amps:
#                     mode = mode.reshape(sysi.Nact,sysi.Nact)

#                     sysi.add_dm(amp*mode)

#                     psf = sysi.calc_psf()
#                     wavefront = psf.wavefront
#                     response += amp*wavefront/np.var(amps)

#                     sysi.add_dm(-amp*mode)

#                 if display:
#                     misc.myimshow2(cp.abs(response), cp.angle(response))

#                 response = response.flatten().get()[dark_mask.flatten()]

#             else:
#                 response = np.zeros((sysi.npsf, sysi.npsf), dtype=np.complex128).flatten()[dark_mask.flatten()]

#             responses.append(np.concatenate((response.real, response.imag)))

#             print('\tCalculated response for mode {:d}/{:d}. Elapsed time={:.3f} sec.'.format(i+1,num_modes,time.time()-start))
        
#     jacobian = np.array(responses).T
    
#     for i in range(len(wavelengths)):
#         jac_new = jac[:,:sysi.Nact**2] if i==0 else np.concatenate((jac_new, jac[:,i*sysi.Nact**2:(i+1)*sysi.Nact**2]), axis=0)
        
#     print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
#     return jac_new

# def run_efc_perfect(sysi, 
#                     wavelengths, 
#                     jac, 
#                     reg_fun,
#                     reg_conds,
#                     dark_mask, 
#                     Imax_unocc,
#                     efc_loop_gain=0.5, 
#                     iterations=5, 
#                     display_all=False, 
#                     display_current=True,
#                     plot_sms=True):
#     # This function is only for running EFC simulations
#     print('Beginning closed-loop EFC simulation.')    
#     commands = []
#     images = []
    
#     start = time.time()
    
#     jac = cp.array(jac) if isinstance(jac, np.ndarray) else jac
    
#     U, s, V = cp.linalg.svd(jac, full_matrices=False)
#     alpha2 = cp.max( cp.diag( cp.real( jac.conj().T @ jac ) ) )
#     print('Max singular value squared:\t', s.max()**2)
#     print('alpha^2:\t\t\t', alpha2) 
    
#     N_DH = dark_mask.sum()
    
#     dm_ref = sysi.get_dm()
#     dm_command = np.zeros((sysi.Nact, sysi.Nact)) 
#     print()
#     for i in range(iterations+1):
#         print('\tRunning iteration {:d}/{:d}.'.format(i, iterations))
        
#         if i==0 or i in reg_conds[0]:
#             reg_cond_ind = np.argwhere(i==reg_conds[0])[0][0]
#             reg_cond = reg_conds[1, reg_cond_ind]
#             print('\tComputing EFC matrix via ' + reg_fun.__name__ + ' with condition value {:.2e}'.format(reg_cond))
#             efc_matrix = reg_fun(jac, reg_cond).get()
        
#         sysi.set_dm(dm_ref + dm_command) 
        
#         psf_bb = 0
#         electric_fields = [] # contains the e-field for each discrete wavelength
#         for wavelength in wavelengths:
#             sysi.wavelength = wavelength
#             psf = sysi.calc_psf()
#             electric_fields.append(psf.wavefront[dark_mask].get())
#             psf_bb += psf.intensity.get()
            
#         commands.append(sysi.get_dm())
#         images.append(copy.copy(psf_bb))
        
#         for j in range(len(wavelengths)):
#             xnew = np.concatenate( (electric_fields[j].real, electric_fields[j].imag) )
#             x = xnew if j==0 else np.concatenate( (x,xnew) )
#         del_dm = efc_matrix.dot(x).reshape(sysi.Nact,sysi.Nact)
        
#         dm_command -= efc_loop_gain * del_dm
        
#         if display_current or display_all:
#             if not display_all: clear_output(wait=True)
                
#             fig,ax = misc.myimshow2(commands[i], images[i], 
#                                         'DM', 'Image: Iteration {:d}'.format(i),
#                                         lognorm2=True, vmin2=1e-12,
#                                         return_fig=True, display_fig=True)
#             if plot_sms:
#                 sms_fig = utils.sms(U, s, alpha2, cp.array(x), N_DH, Imax_unocc, i)
        
#     print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
#     return commands, images





