# loads a backend for calculations, basically the GPU
from .math_module import xp

from . import utils
from . import imshows

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

def build_jacobian(sysi, epsilon, 
                   dark_mask,
                   plot=False,
                  ):
    start = time.time()
    
    amps = np.linspace(-epsilon, epsilon, 2) # for generating a negative and positive actuator poke
    
    dm_mask = sysi.dm_mask.flatten()
    dm_bad_act_mask = sysi.dm_bad_act_mask.flatten()
    
    Nacts = int(dm_mask.sum()) # number of actuators in the pupil
    Ndh = int(dark_mask.sum()) # number of actuators in the DH
    
    num_modes = sysi.Nact**2
    modes = np.eye(num_modes) # each column in this matrix represents a vectorized DM shape where one actuator has been poked
    
    responses = xp.zeros((2*Ndh, Nacts))
    count = 0
    print('Calculating Jacobian: ')
    for i in range(num_modes):
        if dm_mask[i]:
            response = 0
            # Only calculate the jacobian if the actuator is good
            # otherwise it's just left as zero.
            if dm_bad_act_mask[i]:
                for amp in amps:
                    mode = modes[i].reshape(sysi.Nact,sysi.Nact)
                    # adds the poke
                    sysi.add_dm(amp*mode)
                    # calculate wavefront at the focal plane
                    wavefront = sysi.calc_psf()
                    # The next line of code comes from hcipi's tutorial and is quite confusing.
                    # intuitively, the response from each poke would be used to calculate the
                    # "slope", (wavefront1(-ive poke and hence -ive E-field)-wavefront2(+ive poke and +ive e-field))/2*poke amplitude
                    # However, this means making sure the poking is done in the correct order.
                    # The code below multiplies by the poke size and divides by the variance of
                    # the pokes, which is always going to be the modulus of the amplitude for
                    # equal size pokes, and thus compensates for the signs and allows you to 
                    # add the responses.

                    # The factor of two may be missing, and Kian is looking at this.
                    response += amp*wavefront.flatten()/np.var(amps) # why divide by the variance
                    #removes the poke
                    sysi.add_dm(-amp*mode)
                
                responses[::2,count] = response[dark_mask.ravel()].real
                responses[1::2,count] = response[dark_mask.ravel()].imag
            
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
#                     reg_fun,
#                     reg_conds,
                    dark_mask, 
                    Imax_unocc=1,
                    efc_loop_gain=0.5, 
                    iterations=5, 
                    plot_all=False, 
                    plot_current=True,
                    plot_sms=True,
                    plot_radial_contrast=True):

    # This method is called "perfect" because it assumes that 
    # there is perfect knowledge of the electric field.
    
    # This function is only for running EFC simulations
    print('Beginning closed-loop EFC simulation.')    
    commands = []
    efields = []
    
    start = time.time()
    
    # The Sidick et al 2017 paper uses the matrices determined 
    # from running SVD on the jacobian to calculate the 
    # Singular Mode Spectrum (see section 4)
    # This is used in evaluating how well EFC is performing but is
    # not actually used in the EFC calculation itself.

    # Running SVD on the Jacobian essentially derives an orthonormal
    # set of eigenvectors and scaling factors, which are the modes
    # and their relative powers, that are then used
    # to represent the available performance of the control system.
    U, s, V = xp.linalg.svd(jac, full_matrices=False)
    
    # Calculates alpha2 (see notes in utils.beta_reg)
    # This isn't using beta regularization- why? HELP
    # 
    # Coding note:
    # If this is a time consuming calculation then I think 
    # the SVD can be moved out of this function and loop.
    # The max singular value gets calculated externally then
    # the alpha value could get calculated externally as well
    #  
    alpha2 = xp.max( xp.diag( xp.real( jac.conj().T @ jac ) ) )
    # Print max singular value squared, but why do you care?
    print(f'Max singular value squared:\t: {s.max()**2:1.3e}')
    print(f'alpha^2:\t\t\t: {alpha2:1.3e}') 
    
    Ndh = int(dark_mask.sum())
    
    dm_mask = sysi.dm_mask.flatten()

    dm_ref = sysi.get_dm()
    dm_command = np.zeros((sysi.Nact, sysi.Nact)) 
    print()
    for i in range(iterations+1):
        print('\tRunning iteration {:d}/{:d}.'.format(i, iterations))
        sysi.set_dm(dm_ref + dm_command)

        # calculate electric field (complex)
        # this line is the only difference between this and pairwise probing
        electric_field = sysi.calc_psf()

        commands.append(sysi.get_dm())
        efields.append(copy.copy(electric_field))

        efield_ri = xp.zeros(2*Ndh)
        # Fill even indices with real numbers of the E-field
        efield_ri[::2] = electric_field[dark_mask].real
        # Fill odd indices with odd numbers of the E-field
        efield_ri[1::2] = electric_field[dark_mask].imag
        # Calculate how to move the DM
        del_dm = -control_matrix.dot(efield_ri)
        # Map back from flattened array to dm coordinates
        del_dm = xp.array(utils.map_acts_to_dm(utils.ensure_np_array(del_dm), dm_mask))

        # exclude movements of bad actuators
        del_dm *= sysi.dm_bad_act_mask
        # print(f'{del_dm.shape=},{type(del_dm)}')
        # print(f'{sysi.dm_bad_act_mask.shape=},{type(sysi.dm_bad_act_mask)}')

        # Calculate new dm command for the next iteration
        # multiples by the gain for convergence stability
        dm_command += efc_loop_gain * utils.ensure_np_array(del_dm)

        if plot_current or plot_all:

            imshows.imshow3(dm_command, xp.abs(efields[i])**2, dm_ref,
                            'DM Command', f'Image: Iteration {i:d} E-field','dm_ref',
                            lognorm2=True)
            imshows.imshow3(del_dm, sysi.dm_bad_act_mask, del_dm*sysi.dm_bad_act_mask,
                    'DM Command', 'mask','DM Command*mask',
                    lognorm2=False)

            sms_fig = utils.sms(U, s, alpha2, efield_ri, Ndh, Imax_unocc, i, display=plot_sms)

            if plot_radial_contrast:
                utils.plot_radial_contrast(xp.abs(efields[i])**2, dark_mask, sysi.psf_pixelscale_lamD, nbins=100)
            
            if not plot_all: clear_output(wait=True)

    print(f'Total counts in DH: {np.sum(xp.abs(efields[i])**2 *dark_mask):0.3e}')
    print(f'Mean value in DH: {np.mean(xp.abs(efields[i])**2 *dark_mask):0.3e}')    
    print(f'Contrast in DH: {np.std(xp.abs(efields[i])**2 *dark_mask):0.3e}')

    print(f'{iterations} iterations of EFC completed in {(time.time()-start):.3f} sec.')
    
    return commands, efields, sms_fig

def run_efc_pwp(sysi, 
                pwp_fun,
                pwp_kwargs,
                jac,
                control_matrix,
                dark_mask, 
                Imax_unocc=1,
                efc_loop_gain=0.5, 
                iterations=5, 
                plot_all=False, 
                plot_current=True,
                plot_sms=True,
                plot_radial_contrast=True):
    print('Beginning closed-loop EFC simulation using pair-wise probing (PWP)')
    
    commands = []
    efields = []
    images = []
    
    start=time.time()
    
    U, s, V = np.linalg.svd(jac, full_matrices=False)
    alpha2 = np.max( np.diag( np.real( jac.conj().T @ jac ) ) )
    print('Max singular value squared:\t', s.max()**2)
    print('alpha^2:\t\t\t', alpha2) 
    
    Nmask = int(dark_mask.sum())
    
    dm_mask = sysi.dm_mask.flatten()
    if hasattr(sysi, 'bad_acts'):
        dm_mask[sysi.bad_acts] = False
    
    dm_ref = sysi.get_dm()
    dm_command = np.zeros((sysi.Nact, sysi.Nact)) 
    efield_ri = xp.zeros(2*Nmask)
    for i in range(iterations+1):
        print('\tRunning iteration {:d}/{:d}.'.format(i, iterations))
        sysi.set_dm(dm_ref + dm_command)
        E_est = pwp_fun(sysi, dark_mask, **pwp_kwargs)
        I_est = xp.abs(E_est)**2
        I_exact = sysi.snap()

        rms_est = np.sqrt(np.mean(I_est[dark_mask]**2))
        rms_im = np.sqrt(np.mean(I_exact[dark_mask]**2))
        mf = rms_est/rms_im # measure how well the estimate and image match

        commands.append(sysi.get_dm())
        efields.append(copy.copy(E_est))
        images.append(copy.copy(I_exact))

        efield_ri[::2] = E_est[dark_mask].real
        efield_ri[1::2] = E_est[dark_mask].imag
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
                utils.plot_radial_contrast(images[-1], dark_mask, sysi.psf_pixelscale_lamD, nbins=100)

        
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return commands, efields, images
