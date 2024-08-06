from .math_module import xp
from lina.imshows import imshow1, imshow2, imshow3
import lina.utils as utils

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output
import threading as th
import poppy

def calibrate(I, calib_modes, control_mask, amps=5e-9, plot=False):
    # time.sleep(2)
    Nmodes = calib_modes.shape[0]
    Nmask = int(control_mask.sum())
    if np.isscalar(amps):
        amps = [amps] * Nmodes

    responses = np.zeros((Nmodes, Nmask))
    for i in range(Nmodes):
        mode = calib_modes[i]
        amp = amps[i]

        I.add_dm(amp*mode)
        im_pos = I.stack_locam()
        I.add_dm(-2*amp*mode)
        im_neg = I.stack_locam()
        I.add_dm(amp*mode)

        diff = im_pos - im_neg
        responses[i] = diff[control_mask]/(2*amp)

        if plot:
            imshow3(amp*mode, im_pos, diff, f'Mode {i+1}', 'Absolute Image', 'Difference', cmap1='viridis')

    response_matrix = responses.T

    return response_matrix

def run_model(sysi, static_wfe, ref_im, 
              control_matrix, control_modes, 
              time_series,
              zernike_modes,
              control_mask,
              gain_vector=None,  
              reverse_dm_parity=False,
              return_all=True, 
              plot=False):
    """_summary_

    Parameters
    ----------
    sysi : _type_
        _description_
    ref_im : _type_
        _description_
    control_matrix : _type_
        _description_
    control_modes : _type_
        _description_
    time_series_coeff : _type_
        _description_
    zernike_modes : _type_
        _description_
    plot : bool, optional
        _description_, by default False
    """
    print(f'Starting LLOWFSC control-loop simulation: delta T = {time_series[0][1]-time_series[0][0]:.4e}s')

    Nitr = time_series.shape[1]
    Nc = control_modes.shape[0]
    Nz = zernike_modes.shape[0]
    c_modes = control_modes.reshape(Nc, sysi.Nact**2).T
    z_modes = zernike_modes.reshape(Nz, sysi.npix**2).T
    # print(c_modes.shape, z_modes.shape)

    # prior to the first iteration, compute the initial image the first DM commands will be computed from
    new_wfe = z_modes.dot(time_series[1:,0]).reshape(sysi.npix,sysi.npix)
    sysi.WFE = static_wfe * xp.exp(1j*2*np.pi*new_wfe / sysi.wavelength_c.to_value(u.m))

    sysi.use_llowfsc()
    image = sysi.snap()
    del_im = image - ref_im

    if return_all:
        coro_ims = xp.zeros((Nitr-1, sysi.npsf, sysi.npsf))
        llowfsc_ims = xp.zeros((Nitr-1, sysi.nllowfsc, sysi.nllowfsc))

    for i in range(Nitr-1):
        # apply the new wavefront for the current iteration
        new_wfe = z_modes.dot(time_series[1:,i+1]).reshape(sysi.npix,sysi.npix)
        sysi.WFE = static_wfe * xp.exp(1j*2*np.pi*new_wfe / sysi.wavelength_c.to_value(u.m))

        # compute the DM command with the image based on the time delayed wavefront
        modal_coeff = control_matrix.dot(del_im.flatten())
        del_dm_command = -c_modes.dot(modal_coeff).reshape(sysi.Nact,sysi.Nact)
        if reverse_dm_parity:
            del_dm_command = xp.rot90(xp.rot90(del_dm_command))
        sysi.add_dm(del_dm_command/2)
        
        est_opd = xp.rot90(xp.rot90(z_modes.dot(modal_coeff).reshape(sysi.npix,sysi.npix)))
        est_residuals = new_wfe - est_opd
    
        # compute the coronagraphic image after applying the time delayed correction
        sysi.use_llowfsc(False)
        coro_im = sysi.snap()

        # compute the new LLOWFSC image to be used on the next iteration
        sysi.use_llowfsc()
        image = sysi.snap()
        del_im = image - ref_im

        if return_all:
            llowfsc_ims[i] = copy.copy(image)
            coro_ims[i] = copy.copy(coro_im)

        if plot:
            rms_wfe = xp.sqrt(xp.mean(xp.square(new_wfe[sysi.APMASK])))
            rms_est_wfe = xp.sqrt(xp.mean(xp.square(est_opd[sysi.APMASK])))
            rms_residual = xp.sqrt(xp.mean(xp.square(est_residuals[sysi.APMASK])))
            imshows.imshow3(new_wfe, est_opd, del_im, 
                            f'Current WFE: {rms_wfe:.2e}\nTime = {time_series[0][i+1]:.3f}s', 
                            f'Estimated WFE: {rms_est_wfe:.2e}',
                            'Measured Difference Image', 
                            npix1=sysi.npix, npix2=sysi.npix, 
                            vmin1=-20e-9, vmax1=20e-9, 
                            vmin2=-20e-9, vmax2=20e-9, 
                            cmap1='cividis', cmap2='cividis',
                            )
            
            dm_command = sysi.get_dm()
            pv_stroke = xp.max(dm_command) - xp.min(dm_command)
            rms_stroke = xp.sqrt(xp.mean(xp.square(dm_command[sysi.dm_mask])))
            mean_contrast = xp.mean(coro_im[control_mask])
            imshows.imshow3(del_dm_command, dm_command, coro_im, 
                            'Computed DM Correction',
                            f'PV Stroke = {1e9*pv_stroke:.1f}nm\nRMS Stroke = {1e9*rms_stroke:.1f}nm', 
                            f'Coronagraphic Image:\nMean Contrast = {mean_contrast:.2e}', 
                            cmap1='viridis', cmap2='viridis', cmap3='magma', 
                            lognorm3=True, vmin3=1e-11, pxscl3=sysi.psf_pixelscale_lamD, 
                            )
    if return_all:
        return coro_ims, llowfsc_ims


def run_bb_model(sysi,
                 static_wfe, 
                 ref_im, 
                 control_matrix, 
                 control_modes, 
                 wfe_time_series,
                 wfe_modes,
                 gain=1/2,  
                 thresh=0,
                 reverse_dm_parity=False,
                 return_all=True, 
                 plot=False,
                 plot_all=False):
    print(f'Starting LLOWFSC control-loop simulation: delta T = {wfe_time_series[0][1]-wfe_time_series[0][0]:.4e}s')

    Nitr = wfe_time_series.shape[1]
    control_modes = xp.moveaxis(control_modes, 0, -1)
    wfe_modes = xp.moveaxis(wfe_modes, 0, -1)

    llowfsc_ims = xp.zeros((Nitr-1, sysi.nllowfsc, sysi.nllowfsc))
    llowfsc_commands = xp.zeros((Nitr-1, sysi.Nact, sysi.Nact))

    # Prior to the first iteration, compute the initial image the first DM commands will be computed from
    new_wfe = wfe_modes.dot(wfe_time_series[1:,0]).reshape(sysi.npix,sysi.npix)
    sysi.WFE = static_wfe * xp.exp(1j*2*np.pi*new_wfe / sysi.wavelength_c.to_value(u.m))

    image = sysi.snap()
    del_im = image - ref_im

    est_wfe = 0.0
    total_coeff = 0.0
    for i in range(Nitr-1):
        # apply the new wavefront for the current iteration
        new_wfe = wfe_modes.dot(wfe_time_series[1:,i+1]).reshape(sysi.npix,sysi.npix)
        total_wfe = static_wfe * xp.exp(1j*2*np.pi*new_wfe / sysi.wavelength_c.to_value(u.m))
        sysi.set_actor_attr('WFE', total_wfe)

        # compute the DM command with the image based on the time delayed wavefront
        modal_coeff = control_matrix.dot(del_im.flatten())
        modal_coeff *= xp.abs(modal_coeff) >= thresh
        modal_coeff *= gain
        del_dm_command = -control_modes.dot(modal_coeff).reshape(sysi.Nact,sysi.Nact)
        if reverse_dm_parity: del_dm_command = xp.rot90(xp.rot90(del_dm_command))
        sysi.add_dm(del_dm_command)

        # est_wfe += wfe_modes.dot(modal_coeff)
        total_coeff += modal_coeff
        total_est_wfe = 2*wfe_modes.dot(total_coeff).reshape(sysi.npix,sysi.npix)

        # compute the new LLOWFSC image to be used on the next iteration
        image = sysi.snap()
        del_im = image - ref_im

        llowfsc_ims[i] = copy.copy(image)
        llowfsc_commands[i] = copy.copy(sysi.get_dm())

        if plot:
            diff = new_wfe - total_est_wfe
            rms_wfe = xp.sqrt(xp.mean(xp.square(new_wfe[sysi.getattr('APMASK')])))
            # rms_est_wfe = xp.sqrt(xp.mean(xp.square(est_wfe[sysi.getattr('APMASK')])))
            rms_est_wfe = xp.sqrt(xp.mean(xp.square(total_est_wfe[sysi.getattr('APMASK')])))
            rms_diff = xp.sqrt(xp.mean(xp.square(diff[sysi.getattr('APMASK')])))
            imshows.imshow3(new_wfe, total_est_wfe, new_wfe - total_est_wfe, 
                            f'Time = {wfe_time_series[0][i+1]:.3f}s\nCurrent WFE: {rms_wfe:.2e}', 
                            f'Estimated WFE: {rms_est_wfe:.2e}',
                            f'Difference: {rms_diff:.2e}', 
                            npix1=sysi.npix, npix3=sysi.npix, 
                            vmin1=-1.5*rms_wfe, vmax1=1.5*rms_wfe, 
                            vmin2=-1.5*rms_wfe, vmax2=1.5*rms_wfe, 
                            vmin3=-1.5*rms_wfe, vmax3=1.5*rms_wfe, 
                            cmap1='cividis', cmap2='cividis', cmap3='cividis',
                            )
            
            dm_command = sysi.get_dm()
            pv_stroke = xp.max(dm_command) - xp.min(dm_command)
            rms_stroke = xp.sqrt(xp.mean(xp.square(dm_command[sysi.dm_mask])))
            imshows.imshow3(del_im, del_dm_command, dm_command, 
                            'Measured Difference Image', 
                            'Computed DM Correction',
                            f'PV Stroke = {1e9*pv_stroke:.1f}nm\nRMS Stroke = {1e9*rms_stroke:.1f}nm', 
                            cmap1='magma', cmap2='viridis', cmap3='viridis',
                            )
            
            if not plot_all:
                clear_output(wait=True)
            
    return llowfsc_ims, llowfsc_commands

def run_llowfsc_iteration(I,
                          ref_im, 
                          control_matrix, 
                          control_modes,
                          control_mask, 
                          gain=1/2,
                          thresh=0,
                          plot=False,
                          clear=False,
                          ):

    image = I.snap_locam()
    del_im = image - ref_im

    # compute the DM command with the image based on the time delayed wavefront
    modal_coeff = control_matrix.dot(del_im[control_mask])
    modal_coeff *= np.abs(modal_coeff) >= thresh
    modal_coeff *= gain
    del_dm_command = -control_modes.dot(modal_coeff).reshape(I.Nact,I.Nact)
    # if reverse_dm_parity: del_dm_correction = xp.rot90(xp.rot90(del_dm_correction))
    I.add_dm(del_dm_command)

    if plot:
        dm_command = I.get_dm()
        pv_stroke = xp.max(dm_command) - xp.min(dm_command)
        rms_stroke = xp.sqrt(xp.mean(xp.square(dm_command[I.dm_mask])))
        imshow3(del_im, del_dm_command, dm_command, 
                'Measured Difference Image', 
                'Computed DM Correction',
                f'PV Stroke = {1e9*pv_stroke:.1f}nm\nRMS Stroke = {1e9*rms_stroke:.1f}nm', 
                cmap1='magma', cmap2='viridis', cmap3='viridis',
                )
        if clear: clear_output(wait=True)

class Process(th.Timer):  
    def run(self):  
        while not self.finished.wait(self.interval):  
            self.function(*self.args, **self.kwargs)

# process = Repeat(0.1, print, ['Repeating']) 
# process.start()
# time.sleep(5)
# process.cancel()

