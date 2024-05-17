from .math_module import xp
from . import utils, scc
from . import imshows

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

import poppy

def calibrate(sysi, 
              calibration_modes, calibration_amp,
              control_mask, 
              plot=False,
              ):
    """
    This function will generate the response matrix for the LLOWFSC system 
    using a central difference approximation. 
    """

    nmodes = calibration_modes.shape[0]

    responses = xp.zeros((nmodes, sysi.nllowfsc**2))
    for i,mode in enumerate(calibration_modes):
        sysi.set_dm(calibration_amp*mode)
        im_pos = sysi.snap()
        sysi.add_dm(-calibration_amp*mode)
        
        sysi.set_dm(-calibration_amp*mode)
        im_neg = sysi.snap()
        sysi.add_dm(calibration_amp*mode)

        diff = im_pos - im_neg
        responses[i] = diff.flatten()/(2*calibration_amp)

        if plot:
            imshows.imshow3(calibration_amp*mode, im_pos, diff, 
                            f'Calibration Mode {i+1}', 'Absolute Image', 'Difference', 
                            pxscl2=sysi.llowfsc_pixelscale.to(u.mm/u.pix), 
                            pxscl3=sysi.llowfsc_pixelscale.to(u.mm/u.pix), 
                            cmap1='viridis')

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


def run(sysi, ref_im, control_matrix, control_modes, time_series_coeff, zernike_modes,
             lyot_stop=None, 
             reverse_dm_parity=False,
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
    Nitr = time_series_coeff.shape[1]
    Nc = control_modes.shape[0]
    Nz = zernike_modes.shape[0]
    c_modes = control_modes.reshape(Nc, sysi.Nact**2).T
    z_modes = zernike_modes.reshape(Nz, sysi.npix**2).T
    print(c_modes.shape, z_modes.shape)

    if lyot_stop is None:
        lyot_diam = 8.6*u.mm # dont make this hardcoded
        lyot_stop = poppy.CircularAperture(name='Lyot Stop', radius=lyot_diam/2.0)
        wfs_lyot_stop = poppy.InverseTransmission(lyot_stop)

    prev_wfe = xp.zeros((sysi.npix, sysi.npix))
    for i in range(Nitr):
        print(1)
        new_wfe = z_modes.dot(time_series_coeff[:,i]).reshape(sysi.npix,sysi.npix)

        llowfsc_im = sysi.snap_llowfsc()
        del_im = llowfsc_im - ref_im
        
        modal_coeff = 2*control_matrix.dot(del_im.flatten())
        del_dm_command = -c_modes.dot(modal_coeff).reshape(sysi.Nact,sysi.Nact)
        if reverse_dm_parity:
            del_dm_command = xp.rot90(xp.rot90(del_dm_command))
        sysi.add_dm(del_dm_command/2)
        
        est_abs = xp.rot90(xp.rot90(z_modes.dot(modal_coeff).reshape(sysi.npix,sysi.npix)))

        coro_im = sysi.snap()

        if plot:
            rms_wfe = xp.sqrt(xp.mean(xp.square(new_wfe[sysi.pupil_mask])))
            rms_est_wfe = xp.sqrt(xp.mean(xp.square(est_abs[sysi.pupil_mask])))
            rms_residual = xp.sqrt(xp.mean(xp.square(est_residuals[sysi.pupil_mask])))
            imshows.imshow3(new_wfe, est_abs, actual_abs,  
                            f'Current WFE: {rms_wfe:.2e}', 
                            f'Estimated WFE: {rms_est_wfe:.2e}',
                            f'Estimated Residual WFE: {rms_residual:.2e}',
                            npix1=sysi.npix, npix2=sysi.npix, npix3=sysi.npix,
                            vmin1=-20e-9, vmax1=20e-9, vmin2=-20e-9, vmax2=20e-9, vmin3=-20e-9, vmax3=20e-9)
            
            dm_command = sysi.get_dm()
            pv_stroke = xp.max(dm_command) - xp.min(dm_command)
            rms_stroke = xp.sqrt(xp.mean(xp.square(dm_command[sysi.dm_mask])))
            imshows.imshow3(del_im, del_dm_command, coro_im, 
                            'Measured LLOWFSC Image (Difference)', 
                            f'Computed DM Correction:\nPV Stroke = {pv_stroke:.2e}\nRMS Stroke = {rms_stroke:.2e}', 
                            'Coronagraphic Image',
                            cmap2='viridis', 
                            lognorm3=True, vmin3=1e-11, 
                            )
            # imshows.imshow2(est_abs, actual_abs,
            #                 'Estimated WFE', 'True WFE (from model)',
            #                 vmin2=-20e-9, vmax2=20e-9)

        prev_wfe = copy.copy(utils.pad_or_crop(sysi.WFE.opd, sysi.npix))





