from .math_module import xp
from . import utils, scc
from . import imshows

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

def build_jacobian(sysi, 
                   calibration_modes, calibration_amp,
                   control_mask, 
                   plot=False,
                  ):
    """_summary_

    Parameters
    ----------
    sysi : object
        The object of a system interface with methods for DM control and image capture
    calibration_modes : xp.ndarray
        2D array of modes to be calibrated in the Jacobian, shape of (Nmodes, Nact**2)
    calibration_amp : float
        amplitude to be applied to each calibration mode while Jacobian is computed
    control_mask : xp.ndarray
        the boolean mask defining the focal plane pixels in the control region
    plot : bool, optional
        whether of not to plot the RMS response of DM actuators, by default False
    """

    start = time.time()
    
    amps = np.linspace(-calibration_amp, calibration_amp, 2) # for generating a negative and positive actuator poke
    
    Nmodes = calibration_modes.shape[0]
#     Nacts = int(sysi.dm_mask.sum())
    Nmask = int(control_mask.sum())
    
    responses = xp.zeros((2*Nmask, Nmodes))
    print('Calculating Jacobian: ')
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
    
    print()
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    if plot:
        total_response = responses[::2] + 1j*responses[1::2]
        dm_response = total_response.dot(xp.array(calibration_modes))
        dm_response = xp.sqrt(xp.mean(xp.abs(dm_response)**2, axis=0)).reshape(sysi.Nact, sysi.Nact)
        imshows.imshow1(dm_response, lognorm=True, vmin=dm_response.max()*1e-2)

    return responses

def run(sysi, 
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
    """_summary_

    Parameters
    ----------
    sysi : object
        The object of a system interface with methods for DM control and image capture
    jac : xp.ndarray
        The pre-computed Jacobian
    calibration_modes : xp.ndarray
        2D array of modes to be calibrated in the Jacobian, shape of (Nmodes, Nact**2)
    control_matrix : xp.ndarray
        the control matrix used to compute DM commands from the calculated electric field
    control_mask : xp.ndarray
        the boolean mask defining the focal plane pixels in the control region
    loop_gain : float, optional
        how much gain to apply to the computed DM command, 
        should be less than 1 for controller stability, 
        by default 0.5
    leakage : float, optional
       how much leakage to apply to the previously computed DM commands, 
       helps reduce stroke on the DM, 
       by default 0.0
    iterations : int, optional
        number of iterations for which EFC should run, by default 5
    plot_all : bool, optional
        plot results of all iterations, by default False
    plot_current : bool, optional
        plot current iteration results, by default True
    plot_sms : bool, optional
        plot the Singular Mode Spectrum to analyze the dark hole quality,
        FIXME: cite the paper that described this
        by default False
    plot_radial_contrast : bool, optional
        _description_, by default False
    old_images : _type_, optional
        3D array of images from previous iterations, by default None
    old_fields: _type_, optional
        3D array of electric fields estimated or computed during previous iterations, by default None
    old_commands : _type_, optional
        3D array of DM commands computed during previous iterations, by default None

    Returns
    -------
    _type_
        _description_
    """
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
            print('Using model to compute electric field')
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

            if plot_sms:
                sms_fig = sms(U, s, alpha2, efield_ri, Nmask, Imax_unocc, i)

            if plot_radial_contrast:
                utils.plot_radial_contrast(image, control_mask, sysi.psf_pixelscale_lamD, nbins=100)
            
            if not plot_all: clear_output(wait=True)

    metric_images = xp.array(metric_images)
    fields = xp.array(fields)
    dm_commands = xp.array(dm_commands)
    
    if old_images is not None:
        metric_images = xp.concatenate([old_images, metric_images], axis=0)
    if old_fields is not None:
        metric_images = xp.concatenate([old_fields, fields], axis=0)
    if old_commands is not None: 
        dm_commands = xp.concatenate([old_commands, dm_commands], axis=0)
                
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return metric_images, dm_commands

import matplotlib.pyplot as plt

def sms(U, s, alpha2, electric_field, N_DH, 
        Imax_unocc, 
        itr): 
    # jac: system jacobian
    # electric_field: the electric field acquired by estimation or from the model
    
    E_ri = U.conj().T.dot(electric_field)
    SMS = xp.abs(E_ri)**2/(N_DH/2 * Imax_unocc)
    
    Nbox = 31
    box = xp.ones(Nbox)/Nbox
    SMS_smooth = xp.convolve(SMS, box, mode='same')
    
    x = (s**2/alpha2)
    y = SMS_smooth
    
    xmax = float(np.max(x))
    xmin = 1e-10 
    ymax = 1
    ymin = 1e-14
    
    fig = plt.figure(dpi=125, figsize=(6,4))
    plt.loglog(utils.ensure_np_array(x), utils.ensure_np_array(y))
    plt.title('Singular Mode Spectrum: Iteration {:d}'.format(itr))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(r'$(s_{i}/\alpha)^2$: Square of Normalized Singular Values')
    plt.ylabel('SMS')
    plt.grid()
    plt.close()
    display(fig)
    
    return fig