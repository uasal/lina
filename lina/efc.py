from .math_module import xp
from . import utils, scc
from . import imshows

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

def build_jacobian(sysi, epsilon, 
                   control_mask,
                   plot=False,
                  ):
    start = time.time()
    
    amps = np.linspace(-epsilon, epsilon, 2) # for generating a negative and positive actuator poke
    
    dm_mask = sysi.dm_mask.flatten()
    if sysi.bad_acts is not None:
        dm_mask[sysi.bad_acts] = False
    
    Nacts = int(dm_mask.sum())
    Nmask = int(control_mask.sum())
    
    num_modes = sysi.Nact**2
    modes = np.eye(num_modes) # each column in this matrix represents a vectorized DM shape where one actuator has been poked
    
    responses = xp.zeros((2*Nmask, Nacts))
    count = 0
    print('Calculating Jacobian: ')
    for i in range(num_modes):
        if dm_mask[i]:
            response = 0
            for amp in amps:
                mode = modes[i].reshape(sysi.Nact,sysi.Nact)

                sysi.add_dm(amp*mode)
                wavefront = sysi.calc_psf()
                response += amp * wavefront.flatten() / (2*np.var(amps))
                sysi.add_dm(-amp*mode)
            
            responses[::2,count] = response[control_mask.ravel()].real
            responses[1::2,count] = response[control_mask.ravel()].imag
            
            print('\tCalculated response for mode {:d}/{:d}. Elapsed time={:.3f} sec.'.format(count+1, Nacts, time.time()-start), end='')
            print("\r", end="")
            count += 1
        else:
            pass
    print()
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    return responses


def build_jacobian_scc(sysi, epsilon, 
                       control_mask,
                       plot=False,
                       **scc_kwargs,
                      ):
    '''
    This can be done on the actual testbed with individual pokes
    '''
    start = time.time()
    
    amps = np.linspace(-epsilon, epsilon, 2) # for generating a negative and positive actuator poke
    
    dm_mask = sysi.dm_mask.flatten()
#     if hasattr(sysi, 'bad_acts'):
#         dm_mask[sysi.bad_acts] = False
    
    Nacts = int(dm_mask.sum())
    Nmask = int(control_mask.sum())
    
    num_modes = sysi.Nact**2
    modes = np.eye(num_modes) # each column in this matrix represents a vectorized DM shape where one actuator has been poked
    
    responses = xp.zeros((2*Nmask, Nacts))
    count = 0
    print('Calculating Jacobian: ')
    for i in range(num_modes):
        if dm_mask[i]:
            response = 0
            for amp in amps:
                mode = modes[i].reshape(sysi.Nact,sysi.Nact)

                sysi.add_dm(amp*mode)
                wavefront = scc.estimate_coherent(sysi, **scc_kwargs)
                wavefront *= control_mask
                response += amp * wavefront.flatten() / (2*np.var(amps))
                sysi.add_dm(-amp*mode)
            
            responses[::2,count] = response[control_mask.ravel()].real
            responses[1::2,count] = response[control_mask.ravel()].imag
            
            print('\tCalculated response for mode {:d}/{:d}. Elapsed time={:.3f} sec.'.format(count+1, Nacts, time.time()-start), end='')
            print("\r", end="")
            count += 1
        else:
            pass
    print()
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    return responses


def run_efc_perfect(sysi, 
                    jac, 
                    control_matrix,
                    control_mask, 
                    Imax_unocc=1,
                    efc_loop_gain=0.5, 
                    iterations=5, 
                    plot_all=False, 
                    plot_current=True,
                    plot_sms=True,
                    plot_radial_contrast=True):
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
    
    Nmask = int(control_mask.sum())
    
    dm_mask = sysi.dm_mask.flatten()
    if hasattr(sysi, 'bad_acts'):
        dm_mask[sysi.bad_acts] = False
    
    dm_ref = sysi.get_dm()
    dm_command = np.zeros((sysi.Nact, sysi.Nact)) 
    print()
    for i in range(iterations+1):
        print('\tRunning iteration {:d}/{:d}.'.format(i, iterations))
        
#         commands.append(sysi.get_dm())
#         efields.append(copy.copy(electric_field))
#         images.append(copy.copy(image))
        
        electric_field = sysi.calc_psf()
        image = xp.abs(electric_field)**2
        
        commands[i] = sysi.get_dm()
        efields[i] = copy.copy(electric_field)
        images[i] = copy.copy(image)
        
        efield_ri = xp.zeros(2*Nmask)
        efield_ri[::2] = electric_field[control_mask].real
        efield_ri[1::2] = electric_field[control_mask].imag
        del_dm = -control_matrix.dot(efield_ri)

        del_dm = sysi.map_acts_to_dm(del_dm)
        dm_command += efc_loop_gain * utils.ensure_np_array(del_dm)
        
        sysi.set_dm(dm_ref + dm_command)
        
        if plot_current or plot_all:

            imshows.imshow2(commands[i], image, 
                            'DM Command', 'Image: Iteration {:d}'.format(i),
                            cmap1='viridis', lognorm2=True, vmin2=1e-11)

            if plot_sms:
                sms_fig = utils.sms(U, s, alpha2, efield_ri, Nmask, Imax_unocc, i)

            if plot_radial_contrast:
                utils.plot_radial_contrast(xp.abs(efields[i])**2, control_mask, sysi.psf_pixelscale_lamD, nbins=100)
            
            if not plot_all: clear_output(wait=True)
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return images, efields, commands

def run_efc_pwp(sysi, 
                pwp_fun,
                pwp_kwargs,
                jac,
                control_matrix,
                control_mask, 
                Imax_unocc=1,
                efc_loop_gain=0.5, 
                iterations=5, 
                plot_all=False, 
                plot_current=True,
                plot_sms=True,
                plot_radial_contrast=True):
    print('Beginning closed-loop EFC simulation.')
    
    commands = []
    efields = []
    images = []
    
    start=time.time()
    
    U, s, V = np.linalg.svd(jac, full_matrices=False)
    alpha2 = np.max( np.diag( np.real( jac.conj().T @ jac ) ) )
    print('Max singular value squared:\t', s.max()**2)
    print('alpha^2:\t\t\t', alpha2) 
    
    Nmask = int(control_mask.sum())
    
    dm_mask = sysi.dm_mask.flatten()
    if hasattr(sysi, 'bad_acts'):
        dm_mask[sysi.bad_acts] = False
    
    dm_ref = sysi.get_dm()
    dm_command = np.zeros((sysi.Nact, sysi.Nact)) 
    efield_ri = xp.zeros(2*Nmask)
    for i in range(iterations+1):
        print('\tRunning iteration {:d}/{:d}.'.format(i, iterations))
        sysi.set_dm(dm_ref + dm_command)
        E_est = pwp_fun(sysi, control_mask, **pwp_kwargs)
        I_est = xp.abs(E_est)**2
        I_exact = sysi.snap()

        rms_est = np.sqrt(np.mean(I_est[control_mask]**2))
        rms_im = np.sqrt(np.mean(I_exact[control_mask]**2))
        mf = rms_est/rms_im # measure how well the estimate and image match

        commands.append(sysi.get_dm())
        efields.append(copy.copy(E_est))
        images.append(copy.copy(I_exact))

        efield_ri[::2] = E_est[control_mask].real
        efield_ri[1::2] = E_est[control_mask].imag
        del_dm = -control_matrix.dot(efield_ri)

        del_dm = sysi.map_actuators_to_command(del_dm)
        dm_command += efc_loop_gain * del_dm

        if plot_current or plot_all:
            if not plot_all: clear_output(wait=True)

            imshows.imshow3(commands[i], I_est, I_exact, 
                            lognorm2=True, lognorm3=True)

            if plot_sms:
                sms_fig = utils.sms(U, s, alpha2, efield_ri, Nmask, Imax_unocc, i)

            if plot_radial_contrast:
                utils.plot_radial_contrast(images[-1], control_mask, sysi.psf_pixelscale_lamD, nbins=100)

        
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return commands, efields, images


def run_efc_scc(sysi, 
                jac,
                control_matrix,
                control_mask, 
                Imax_unocc=1,
                efc_loop_gain=0.5, 
                iterations=5, 
                plot_all=False, 
                plot_current=True,
                plot_sms=True,
                plot_radial_contrast=True,
                **scc_kwargs,):
    print('Beginning closed-loop EFC simulation.')
    
    commands = []
    efields = []
    images = []
    
    start=time.time()
    
    U, s, V = np.linalg.svd(jac, full_matrices=False)
    alpha2 = np.max( np.diag( np.real( jac.conj().T @ jac ) ) )
    print('Max singular value squared:\t', s.max()**2)
    print('alpha^2:\t\t\t', alpha2) 
    
    Nmask = int(control_mask.sum())
    
    dm_mask = sysi.dm_mask.flatten()
    if hasattr(sysi, 'bad_acts'):
        dm_mask[sysi.bad_acts] = False
    
    dm_ref = sysi.get_dm()
    dm_command = np.zeros((sysi.Nact, sysi.Nact)) 
    efield_ri = xp.zeros(2*Nmask)
    for i in range(iterations+1):
        print('\tRunning iteration {:d}/{:d}.'.format(i, iterations))
        sysi.set_dm(dm_ref + dm_command)
        E_est = scc.estimate_coherent(sysi, **scc_kwargs)
        E_est *= control_mask
        I_est = xp.abs(E_est)**2
        I_exact = sysi.snap()

        rms_est = np.sqrt(np.mean(I_est[control_mask]**2))
        rms_im = np.sqrt(np.mean(I_exact[control_mask]**2))
        mf = rms_est/rms_im # measure how well the estimate and image match

        commands.append(sysi.get_dm())
        efields.append(copy.copy(E_est))
        images.append(copy.copy(I_exact))

        efield_ri[::2] = E_est[control_mask].real
        efield_ri[1::2] = E_est[control_mask].imag
        del_dm = -control_matrix.dot(efield_ri)

        del_dm = sysi.map_actuators_to_command(del_dm)
        dm_command += efc_loop_gain * del_dm

        if plot_current or plot_all:
            if not plot_all: clear_output(wait=True)

            imshows.imshow3(commands[i], I_est, I_exact, 
                            lognorm2=True, lognorm3=True)

            if plot_sms:
                sms_fig = utils.sms(U, s, alpha2, efield_ri, Nmask, Imax_unocc, i)

            if plot_radial_contrast:
                utils.plot_radial_contrast(images[-1], control_mask, sysi.psf_pixelscale_lamD, nbins=100)

        
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return commands, efields, images