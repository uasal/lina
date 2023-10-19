import numpy as np
try:
    import cupy as cp
    xp = cp
except ImportError:
    xp = np
    
import poppy

import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

import imshows
from . import utils


def compute_E_DM(actuators, forward_model, model_params):
    E_DM = forward_model(model_params['actuators'], model_params['pupil'], model_params['wfe'], model_params['fpm'], model_params['lyot'],
                         model_params['inf_matrix'], model_params['inf_pixelscale'],
                         model_params['npix'], model_params['oversample'], Imax_ref=model_params['Imax_ref'])

    return E_DM

def cost_fun(actuators, I_tar_ratio=1/2, eta_00=1,):
    '''
    acts: the actuator vector we want to find an optimal solution for
    I_tar_ratio: ratio for the desired target irradiance based on current integrated irradiance
    eta_00: parameter that Scott understands more
    E_ab: current electric-field estimate
    '''
    E_DM = compute_E_DM(actuators)[control_mask]
    E_DZ = E_ab[control_mask] + E_DM
    I_DZ = jnp.abs(E_DZ.conj().dot(E_DZ))
    I_tar = I_tar_ratio * I_DZ
    J = actuators.T.dot(actuators) + eta_00 * ((I_DZ - I_tar)/I_tar)**2
    return J


def run(sysi, 
        estimation='perfect',
        estimation_params=None,
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
    commands = []
    efields = []
    
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
        sysi.set_dm(dm_ref + dm_command)

        if estimation=='perfect':
            electric_field = sysi.calc_psf()
        elif estimation=='pwp':
            electric_field = estimation_fun(estiamtion_params)
        

        commands.append(sysi.get_dm())
        efields.append(copy.copy(electric_field))

        efield_ri = xp.zeros(2*Ndh, dtype=control_matrix.dtype)
        efield_ri[::2] = electric_field[control_mask].real
        efield_ri[1::2] = electric_field[control_mask].imag
        del_dm = -control_matrix.dot(efield_ri)
        
        del_dm = xp.array(utils.map_acts_to_dm(utils.ensure_np_array(del_dm), dm_mask))
        dm_command += efc_loop_gain * utils.ensure_np_array(del_dm)

        if plot_current or plot_all:

            imshows.imshow2(commands[i], xp.abs(efields[i])**2, 
                            'DM Command', 'Image: Iteration {:d}'.format(i), 
                            cmap1='viridis', lognorm2=True, vmin2=1e-11)

            if plot_sms:
                sms_fig = utils.sms(U, s, alpha2, efield_ri, Nmask, Imax_unocc, i)

            if plot_radial_contrast:
                utils.plot_radial_contrast(xp.abs(efields[i])**2, control_mask, sysi.psf_pixelscale_lamD, nbins=100)
            
            if not plot_all: clear_output(wait=True)
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return commands, efields

def build_adjoint_model():
    
    return


