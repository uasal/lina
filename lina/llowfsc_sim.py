from .math_module import xp, ensure_np_array
import lina.utils as utils
from lina.imshows import imshow1, imshow2, imshow3

import numpy as np
import astropy.units as u
import copy
from IPython.display import display, clear_output
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

def calibrate_without_fsm(I, control_mask, dm_modes, amps=5e-9, plot=False):
    # time.sleep(2)
    Nmask = int(control_mask.sum())
    Nmodes = dm_modes.shape[0]
    if np.isscalar(amps):
        amps = [amps] * Nmodes

    if isinstance(control_mask, np.ndarray):
        responses = np.zeros((Nmodes, Nmask))
    else:
        responses = xp.zeros((Nmodes, Nmask))
    
    start = time.time()
    for i in range(Nmodes):
        amp = amps[i]
        mode = dm_modes[i]

        I.add_dm(amp*mode)
        im_pos = I.snap_locam()
        I.add_dm(-2*amp*mode)
        im_neg = I.snap_locam()
        I.add_dm(amp*mode)

        diff = im_pos - im_neg
        responses[i] = copy.copy(diff)[control_mask]/(2 * amp)
        
        if plot:
            imshow3(amp*mode, im_pos, diff, f'Mode {i+1}', 'Absolute Image', 'Difference', cmap1='viridis')
        
        print(f"\tCalibrated mode {i+1:d}/{dm_modes.shape[0]:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")

    response_matrix = responses.T

    return response_matrix

def calibrate_with_fsm(I, control_mask, dm_modes=None, amps=5e-9, plot=False):
    # time.sleep(2)
    Nmask = int(control_mask.sum())
    Nmodes = dm_modes.shape[0] + 2 if dm_modes is not None else 2
    if np.isscalar(amps):
        amps = [amps] * Nmodes
    
    pupil_mask = utils.create_circ_mask(500, 500, radius=250)
    TTM = utils.create_zernike_modes(pupil_mask, nmodes=2, remove_modes=1)

    responses = xp.zeros((Nmodes, Nmask))
    for i in range(Nmodes):
        amp = amps[i]

        if i==0: # assumes this is the Tip mode so applies it to the FSM
            mode = TTM[0]
            amp_pv = amp / I.tt_pv_to_rms
            amp_as = (np.arctan(amp_pv / I.fsm_beam_diam.to_value(u.m))*u.radian).to(u.arcsec) 
            I.add_fsm(tip=amp_as, tilt=0*u.arcsec)
            im_pos = I.snap_locam()
            I.add_fsm(tip=-2*amp_as, tilt=0*u.arcsec)
            im_neg = I.snap_locam()
            I.add_fsm(tip=amp_as, tilt=0*u.arcsec)
        elif i==1: # assumes this is the Tilt mode so applies it to the FSM
            mode = TTM[1]
            amp_pv = amp / I.tt_pv_to_rms
            amp_as = (np.arctan(amp_pv / I.fsm_beam_diam.to_value(u.m))*u.radian).to(u.arcsec) 
            I.add_fsm(tip=0*u.arcsec, tilt=amp_as)
            im_pos = I.snap_locam()
            I.add_fsm(tip=0*u.arcsec, tilt=-2*amp_as)
            im_neg = I.snap_locam()
            I.add_fsm(tip=0*u.arcsec, tilt=amp_as)
        else: # use DM for all higher order modes
            mode = dm_modes[i-2]
            I.add_dm(amp*mode)
            im_pos = I.snap_locam()
            I.add_dm(-2*amp*mode)
            im_neg = I.snap_locam()
            I.add_dm(amp*mode)

        diff = im_pos - im_neg
        responses[i] = copy.copy(diff)[control_mask]/(2 * amp)

        if plot:
            imshow3(amp*mode, im_pos, diff, f'Mode {i+1}', 'Absolute Image', 'Difference', cmap1='viridis')

    response_matrix = responses.T

    return response_matrix

def run_sim_with_fsm(
    M, 
    static_amp,
    static_opd, 
    ref_im, 
    control_mask, 
    control_matrix, 
    time_series, 
    wfe_modes, 
    dm_modes,
    gain=1/2,  
    leakage=0.0,
    dh_command=xp.zeros((34,34)),
    data_dict={},
    plot=False, 
    plot_all=False,
    sleep=0.25,
    ):
    """_summary_

    Parameters
    ----------
    sysi : _type_
        _description_
    ref_im : _type_
        _description_
    control_matrix : _type_
        _description_
    dm_modes : _type_
        _description_
    time_series_coeff : _type_
        _description_
    zernike_modes : _type_
        _description_
    plot : bool, optional
        _description_, by default False
    """
    Nitr = time_series.shape[1]
    llowfsc_ims = xp.zeros((Nitr, M.nlocam, M.nlocam))
    diff_ims = xp.zeros((Nitr, M.nlocam, M.nlocam))
    fsm_commands = xp.zeros((Nitr, 2))
    coro_ims = xp.zeros((Nitr, M.npsf, M.npsf))
    lo_commands = xp.zeros((Nitr, M.Nact, M.Nact))
    injected_wfes = xp.zeros((Nitr, time_series[1:, 0].shape[0]))

    start = time.time()

    # total_fsm_command = old_fsm_command
    # total_lo_command = old_lo_command
    total_fsm_command = xp.zeros(2) if 'fsm_commands' not in data_dict else data_dict['fsm_commands'][-1]
    total_lo_command = xp.zeros((M.Nact, M.Nact)) if 'lo_commands' not in data_dict else data_dict['lo_commands'][-1]
    for i in range(Nitr):
        if i==0:
            del_fsm_command_rms = xp.zeros(2)
            del_dm_command = xp.zeros((M.Nact, M.Nact))
        else:
            # compute the DM command with the image based on the time delayed wavefront
            modal_coeff = - gain * control_matrix.dot(del_im[control_mask])
            del_fsm_command_rms = modal_coeff[:2]
            del_dm_command = xp.sum( modal_coeff[2:, None, None] * dm_modes, axis=0)
        # print(old_fsm_command, del_fsm_command_rms)
        total_fsm_command = (1-leakage) * total_fsm_command + del_fsm_command_rms
        total_fsm_pv = total_fsm_command / M.tt_pv_to_rms
        total_fsm_as = 1/2 * (np.arctan( ensure_np_array(total_fsm_pv) / M.fsm_beam_diam.to_value(u.m))*u.radian).to(u.arcsec)
        M.set_fsm(tip=total_fsm_as[0], tilt=total_fsm_as[1])

        total_lo_command = (1-leakage) * total_lo_command + del_dm_command
        M.set_dm(dh_command + total_lo_command)

        # apply the new wavefront to simulate a time delay 
        lo_wfe = xp.sum( time_series[1:, i, None, None] * wfe_modes, axis=0)
        M.setattr('AMP', static_amp )
        M.setattr('OPD', static_opd + lo_wfe) 
        locam_im = M.snap_locam()
        coro_im = M.snap()
        del_im = control_mask * (locam_im - ref_im)

        llowfsc_ims[i] = copy.copy(locam_im)
        diff_ims[i] = copy.copy(del_im)
        coro_ims[i] = copy.copy(coro_im)
        fsm_commands[i] = copy.copy(total_fsm_command)
        lo_commands[i] = copy.copy(total_lo_command)
        injected_wfes[i] = copy.copy(time_series[1:, i])
        # print(total_lo_command)
        if plot or plot_all:
            # time.sleep(sleep)
            fig,ax = plt.subplots(nrows=2, ncols=3, figsize=(15,9), dpi=125, gridspec_kw={'width_ratios': [1, 1, 1], })
            fig.suptitle(f'Time: {time_series[0,i]:.3f}s\nLLOWFSC Iteration {i+1:d}', fontsize=16)

            im1 = ax[0,0].imshow(ensure_np_array(locam_im), norm=Normalize(vmin=0), cmap='magma',)
            ax[0,0].set_title(f'LOCAM Image', fontsize=14)
            divider = make_axes_locatable(ax[0,0])
            cax = divider.append_axes("right", size="4%", pad=0.075)
            cbar = fig.colorbar(im1, cax=cax)
            # ax[0].set_position([0, 0.3, 0.25, 0.25]) # [left, bottom, width, height]

            im2 = ax[0,1].imshow( ensure_np_array(del_im), norm=Normalize(), cmap='magma')
            ax[0,1].set_title(f'Difference Image', fontsize=14)
            divider = make_axes_locatable(ax[0,1])
            cax = divider.append_axes("right", size="4%", pad=0.075)
            cbar = fig.colorbar(im2, cax=cax,)

            im3 = ax[0,2].imshow( ensure_np_array(coro_im), norm=LogNorm(vmin=1e-9, vmax=1e-4), cmap='magma')
            ax[0,2].set_title(f'Coro Image', fontsize=14)
            divider = make_axes_locatable(ax[0,2])
            cax = divider.append_axes("right", size="4%", pad=0.075)
            cbar = fig.colorbar(im3, cax=cax,)

            im4 = ax[1,0].imshow(ensure_np_array(lo_wfe), norm=Normalize(), cmap='viridis',)
            ax[1,0].set_title(f'Injected WFE', fontsize=14)
            divider = make_axes_locatable(ax[1,0])
            cax = divider.append_axes("right", size="4%", pad=0.075)
            cbar = fig.colorbar(im4, cax=cax)

            im5 = ax[1,1].imshow(ensure_np_array(M.getattr('FSM')), norm=Normalize(), cmap='viridis',)
            ax[1,1].set_title(f'FSM Command', fontsize=14)
            divider = make_axes_locatable(ax[1,1])
            cax = divider.append_axes("right", size="4%", pad=0.075)
            cbar = fig.colorbar(im5, cax=cax)

            im6 = ax[1,2].imshow(ensure_np_array(total_lo_command), norm=Normalize(), cmap='viridis',)
            ax[1,2].set_title(f'LO DM Command', fontsize=14)
            divider = make_axes_locatable(ax[1,2])
            cax = divider.append_axes("right", size="4%", pad=0.075)
            cbar = fig.colorbar(im6, cax=cax)

            plt.close()
            display(fig)

            if not plot_all: clear_output(wait=True)
        else:
            print(f"\tLLOWFSC running iteration {i+1:d} in {time.time()-start:.3f}s", end='')
            print("\r", end="")

    if 'llowfsc_ims' in data_dict: data_dict['llowfsc_ims'] = xp.concatenate([data_dict['llowfsc_ims'], llowfsc_ims])
    else: data_dict.update({'llowfsc_ims':llowfsc_ims})
    if 'diff_ims' in data_dict: data_dict['diff_ims'] = xp.concatenate([data_dict['diff_ims'], diff_ims])
    else: data_dict.update({'diff_ims':diff_ims})
    if 'coro_ims' in data_dict: data_dict['coro_ims'] = xp.concatenate([data_dict['coro_ims'], coro_ims])
    else: data_dict.update({'coro_ims':coro_ims})
    if 'wfes' in data_dict: data_dict['wfes'] = xp.concatenate([data_dict['wfes'], injected_wfes])
    else: data_dict.update({'wfes':injected_wfes})
    if 'fsm_commands' in data_dict: data_dict['fsm_commands'] = xp.concatenate([data_dict['fsm_commands'], fsm_commands])
    else: data_dict.update({'fsm_commands':fsm_commands})
    if 'lo_commands' in data_dict: data_dict['lo_commands'] = xp.concatenate([data_dict['lo_commands'], lo_commands])
    else: data_dict.update({'lo_commands':lo_commands})
    if 'dh_commands' in data_dict: data_dict['dh_commands'] = xp.concatenate([data_dict['dh_commands'], xp.array([dh_command])])
    else: data_dict.update({'dh_commands':xp.array([dh_command])})
    if 'gains' in data_dict: data_dict['gains'].append(gain)
    else: data_dict.update({'gains':[gain]})
    if 'leakages' in data_dict: data_dict['leakages'].append(leakage)
    else: data_dict.update({'leakages':[leakage]})
    
    return data_dict

def run_sim(M, 
              static_wfe, 
              ref_im, 
              control_mask, 
              control_matrix, 
              time_series, 
              wfe_modes, 
              dm_modes,
              gain=1/2,  
              leakage=0.0,
              dh_command=xp.zeros((34,34)),
              old_lo_command=0.0, 
              plot=False, 
              plot_all=False,
              sleep=None, 
              ):
    """_summary_

    Parameters
    ----------
    sysi : _type_
        _description_
    ref_im : _type_
        _description_
    control_matrix : _type_
        _description_
    dm_modes : _type_
        _description_
    time_series_coeff : _type_
        _description_
    zernike_modes : _type_
        _description_
    plot : bool, optional
        _description_, by default False
    """
    print(f'Starting LLOWFSC control-loop simulation')

    Nitr = time_series.shape[1]
    llowfsc_ims = xp.zeros((Nitr, M.nlocam, M.nlocam))
    diff_ims = xp.zeros((Nitr, M.nlocam, M.nlocam))
    coro_ims = xp.zeros((Nitr, M.npsf, M.npsf))
    lo_commands = xp.zeros((Nitr, M.Nact, M.Nact))
    injected_wfes = xp.zeros((Nitr, time_series[1:, 0].shape[0]))
    
    for i in range(Nitr):
        if sleep is not None: time.sleep(sleep)
        if i==0:
            del_dm_command = xp.zeros_like(M.DM.command)
        else:
            # compute the DM command with the image based on the time delayed wavefront
            modal_coeff = - gain * control_matrix.dot(del_im[control_mask])
            del_dm_command = xp.sum( modal_coeff[:, None, None] * dm_modes, axis=0)
        total_lo_dm = (1-leakage) * old_lo_command + del_dm_command
        M.set_dm(dh_command + total_lo_dm)

        # apply the new wavefront to simulate a time delay 
        lo_wfe = xp.sum( time_series[1:, i, None, None] * wfe_modes, axis=0)
        M.setattr('WFE', static_wfe * xp.exp(1j * 2*np.pi/M.wavelength_c.to_value(u.m) * lo_wfe) )
        locam_im = M.snap_locam()
        coro_im = M.snap()
        del_im = locam_im - ref_im

        llowfsc_ims[i] = copy.copy(locam_im)
        diff_ims[i] = copy.copy(del_im)
        coro_ims[i] = copy.copy(coro_im)
        lo_commands[i] = copy.copy(total_lo_dm)
        injected_wfes[i] = copy.copy(time_series[1:, i])

        old_lo_command = copy.copy(total_lo_dm)

        if plot or plot_all:
            imshow3(locam_im, control_mask*del_im, coro_im, 
                    'LLOWFSC Image', 'Difference Image',
                    cmap1='magma', cmap2='magma',
                    lognorm3=True, vmin3=1e-9, )
            rms_wfe = xp.sqrt(xp.mean(xp.square( lo_wfe[M.APMASK] )))
            vmax_pup = 2*rms_wfe
            pupil_cmap = 'viridis'
            imshow3(lo_wfe, del_dm_command, M.get_dm(), 
                    f'Current WFE: {rms_wfe:.2e}\nTime = {time_series[0][i]:.3f}s', 
                    'LLOWFSC DM Command', 'Total DM Command',
                    vmin1=-vmax_pup, vmax1=vmax_pup, 
                    cmap1=pupil_cmap, cmap2=pupil_cmap, cmap3=pupil_cmap,
                    )
            
            if not plot_all: clear_output(wait=True)

    sim_dict = {
        'llowfsc_ims':llowfsc_ims,
        'diff_ims':diff_ims, 
        'injected_wfes':injected_wfes,
        'llowfsc_ref':ref_im,
        'wfe_modes':wfe_modes, 
        'coro_ims':coro_ims,
        'lo_commands':lo_commands,
    }
    
    return sim_dict

def plot_data(data):
    ims = ensure_np_array( xp.array(data['images']) ) 
    control_mask = ensure_np_array( data['control_mask'] )
    # print(type(control_mask))
    Nitr = ims.shape[0]
    npsf = ims.shape[1]
    psf_pixelscale_lamD = data['pixelscale']

    mean_nis = np.mean(ims[:,control_mask], axis=1)
    ibest = np.argmin(mean_nis)
    ref_im = ensure_np_array(data['images'][0])
    best_im = ensure_np_array(data['images'][ibest])

    fig,ax = plt.subplots(nrows=2, ncols=3, figsize=(15,10), dpi=125, gridspec_kw={'width_ratios': [1, 1, 1.35], })

    im1 = ax[0].imshow(ref_im, norm=LogNorm(vmax=vmax, vmin=vmin), cmap='magma', extent=extent)
    ax[0].set_title(f'Reference Image:\nMean Contrast = {mean_nis[0]:.2e}', fontsize=14)
    # divider = make_axes_locatable(ax[0])
    # cax = divider.append_axes("right", size="4%", pad=0.075)
    # cbar = fig.colorbar(im1, cax=cax)
    # cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)
    ax[0].set_position([0, 0.3, 0.25, 0.25]) # [left, bottom, width, height]

    im2 = ax[1].imshow( best_im, norm=LogNorm(vmax=vmax, vmin=vmin), cmap='magma', extent=extent)
    ax[1].set_title(f'Best Iteration:\nMean Contrast = {mean_nis[ibest]:.2e}', fontsize=14)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im2, cax=cax,)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)
    # ax[1].set_position([0.212, 0.3, 0.25, 0.25])

    ax[0].set_ylabel('Y [$\lambda/D$]', fontsize=12, labelpad=-5)
    ax[0].set_xlabel('X [$\lambda/D$]', fontsize=12, labelpad=5)
    ax[1].set_xlabel('X [$\lambda/D$]', fontsize=12, labelpad=5)

    ax[2].set_title('Mean Contrast per Iteration', fontsize=14)
    ax[2].semilogy(mean_nis, label='3.6% Bandpass')
    ax[2].grid()
    ax[2].set_xlabel('Iteration Number', fontsize=12, )
    ax[2].set_ylabel('Mean Contrast', fontsize=14, labelpad=1)
    ax[2].set_ylim([vmin, vmax])
    ax[2].set_xticks(np.arange(0,Nitr,2))
    ax[2].set_position([0.525, 0.3, 0.25, 0.25])




