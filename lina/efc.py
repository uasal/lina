from .math_module import xp, ensure_np_array
import lina.utils as utils
import lina.imshows as imshows

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

def compute_jacobian(M,
                     calib_modes,
                     control_mask,
                     amp=1e-9,
                     plot_responses=True,
                     ):
    Nmodes = calib_modes.shape[0]
    Nmask = int(control_mask.sum())
    jac = xp.zeros((2*Nmask, Nmodes))

    for i in range(Nmodes):
        E_pos = M.forward(amp*calib_modes[i][M.dm_mask], use_wfe=True, use_vortex=True)[control_mask]
        E_neg = M.forward(-amp*calib_modes[i][M.dm_mask], use_wfe=True, use_vortex=True)[control_mask]
        response = (E_pos - E_neg)/(2*amp)
        jac[::2,i] = xp.real(response)
        jac[1::2,i] = xp.imag(response)

    if plot_responses:
        dm_rms = xp.sqrt(xp.mean(xp.square(jac.dot(calib_modes)), axis=0))
        dm_rms = dm_rms.reshape(M.Nact, M.Nact) / xp.max(dm_rms)
        imshows.imshow1(dm_rms, 'DM RMS Actuator Responses', lognorm=True, vmin=1e-2)

    return jac
    
def calibrate(sysi, 
              calibration_modes, calibration_amp,
              control_mask, 
              scc_fun=None, scc_params=None,
              plot=False,
              return_all=False, 
              ):
    """
    This function will compute the Jacobian for EFC using either the system model 
    or the SCC estimation function. If SCC is used, this function can be used with a real instrument. 

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
    
    Nmodes = calibration_modes.shape[0]
    Nmask = int(control_mask.sum())
    
    response_matrix = xp.zeros((2*Nmask, Nmodes), dtype=xp.float64)
    if return_all: response_cube = xp.zeros((sysi.npsf**2, 2, Nmodes))
    print('Calculating Jacobian: ')
    for i,dm_mode in enumerate(calibration_modes):
        response = 0
        for s in [-1, 1]:
            # Add the mode to the DMs
            sysi.add_dm(s * calibration_amp * dm_mode.reshape(sysi.Nact,sysi.Nact))
            # Compute/Measure reponse with model or SCC
            efield = sysi.calc_wf() if scc_fun is None else scc_fun(sysi, **scc_params)
            response += s * efield / (2 * calibration_amp)
            # Remove the mode form the DMs
            sysi.add_dm(-s * calibration_amp * dm_mode.reshape(sysi.Nact,sysi.Nact))

        response_matrix[::2,i] = response[control_mask].real
        response_matrix[1::2,i] = response[control_mask].imag
        if return_all:
            response_cube[:,0,i] = xp.real(response.ravel())
            response_cube[:,1,i] = xp.imag(response.ravel())

        print('\tCalculated response for mode {:d}/{:d}. Elapsed time={:.3f} sec.'.format(i+1, Nmodes, time.time()-start), end='')
        print("\r", end="")
    
    print()
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    if plot:
        dm_responses = response_matrix[::2] + 1j*response_matrix[1::2]
        dm_responses = dm_responses.dot(xp.array(calibration_modes.reshape(Nmodes, -1)))
        dm_response_map = xp.sqrt(xp.mean(xp.abs(dm_responses)**2, axis=0)).reshape(sysi.Nact, sysi.Nact)
        dm_response_map /= xp.max(dm_response_map)
        imshows.imshow1(dm_response_map, lognorm=True, vmin=1e-2)

    if return_all:
        return response_matrix, response_cube
    else:
        return response_matrix

def run(sysi, 
        control_matrix,
        calibration_modes,
        control_mask, 
        est_fun=None,
        est_params=None,
        loop_gain=0.5, 
        leakage=0.0,
        iterations=5, 
        plot_all=False, 
        plot_current=True,
        plot_sms=False,
        plot_radial_contrast=False,
        all_ims=None,
        all_efs=None,
        all_commands=None,
        ):
    print('Beginning closed-loop EFC.')    
    start = time.time()
    calibration_modes = xp.array(calibration_modes)
    Nmodes = calibration_modes.shape[0]
    Nmask = int(control_mask.sum())
    starting_itr = len(all_ims)
    if len(all_commands)>0:
        total_command = copy.copy(all_commands[-1])
    else:
        total_command = xp.zeros((sysi.Nact,sysi.Nact))
    efield_ri = xp.zeros(2*Nmask)
    for i in range(iterations):
        print(f"\tClosed-loop iteration {i+1+starting_itr} / {iterations+starting_itr}")
        
        if est_fun is not None and est_params is not None:
            print(f'Using {est_fun.__name__} to estimate electric field')
            electric_field = est_fun(sysi, **est_params)
        else:
            print('Using model to compute electric field')
            electric_field = sysi.calc_wf() # no PWP, just use model
        
        efield_ri[::2] = electric_field[control_mask].real
        efield_ri[1::2] = electric_field[control_mask].imag

        modal_coeff = -control_matrix.dot(efield_ri)
        # del_command = calibration_modes.T.dot(modal_coeff).reshape(sysi.Nact,sysi.Nact)
        del_command = calibration_modes.reshape(Nmodes, -1).T.dot(modal_coeff).reshape(sysi.Nact,sysi.Nact)
        total_command = (1.0-leakage)*total_command + loop_gain*del_command
        sysi.set_dm(total_command)

        image = sysi.snap()

        all_ims.append(copy.copy(image))
        all_efs.append(copy.copy(electric_field))
        all_commands.append(copy.copy(total_command))

        if plot_current: 
            if not plot_all: clear_output(wait=True)
            mean_ni = xp.mean(image[control_mask])
            imshows.imshow3(del_command, total_command, image, 
                    f'Iteration {starting_itr + i + 1:d}: $\delta$DM', 
                    'Total DM Command', 
                    f'Image\nMean NI = {mean_ni:.3e}',
                    cmap1='viridis', cmap2='viridis', 
                    pxscl3=sysi.psf_pixelscale_lamD, lognorm3=True, vmin3=1e-9)
            
            # if plot_sms:
            #     sms_fig = sms(U, s, alpha2, efield_ri, Nmask, i)

            if plot_radial_contrast:
                utils.plot_radial_contrast(image, control_mask, sysi.psf_pixelscale_lamD, nbins=100)
            
            if not plot_all: clear_output(wait=True)
                
    print(f'EFC completed in {time.time()-start:.3f} sec.')
    
    return all_ims, all_efs, all_commands

import matplotlib.pyplot as plt

def sms(U, s, alpha2, electric_field, N_DH, itr): 
    # electric_field: the electric field acquired by estimation or from the model
    Imax_unocc = 1 # assuming that all the images are already normalized

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