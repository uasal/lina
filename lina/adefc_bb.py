from .math_module import xp, _scipy, ensure_np_array
import lina.utils as utils
from lina.imshows import imshow1, imshow2, imshow3
import lina.adefc as adefc

import numpy as np
from scipy.optimize import minimize
import time
import copy

def calc_wfs(I, waves, control_mask, plot=False):
    Nwaves = len(waves)
    E_abs = xp.zeros((Nwaves, I.npsf, I.npsf), dtype=xp.complex128)
    for i in range(Nwaves):
        wavelength = waves[i]
        E_abs[i] = I.calc_wf(wavelength=wavelength) * control_mask
        if plot: imshow2(xp.abs(E_abs[i])**2, xp.angle(E_abs[i])*control_mask, lognorm1=True, cmap2='twilight')

    return E_abs

def run_pwp_bb(I, 
                M, 
                current_acts, 
                control_mask, 
                probes, probe_amp,
                bandpasses, 
                reg_cond=1e-3,
                plot=False,
                plot_est=False,
                ):
    
    Nbps = bandpasses.shape[0]
    Nwaves_per_bp = bandpasses.shape[1]

    E_ests = xp.zeros((Nbps, I.npsf, I.npsf))
    for i in range(Nbps):
        I.setattr('waves', bandpasses[i])
        M.setattr('wavelength', bandpasses[i, Nwaves_per_bp//2]) 
        E_est_mono = adefc.run_pwp(I, M, current_acts, control_mask, probes, probe_amp, reg_cond, plot=plot, plot_est=plot_est)
        E_ests[i] = copy.copy(E_est_mono)

    I.setattr('waves', bandpasses.flatten()) # reset the wavelengths of I to the full bandpass
    M.setattr('wavelength', M.wavelength_c)

    return E_ests

def run(I, 
        M, 
        val_and_grad_bb,
        control_mask,
        bandpasses, 
        data,
        pwp_params=None,
        Nitr=3, 
        reg_cond=1e-2,
        bfgs_tol=1e-3,
        bfgs_opts=None,
        gain=0.5, 
        leakage=0.0, 
        vmin=1e-9, 
        ):

    Nbps = bandpasses.shape[0]
    Nwaves_per_bp = bandpasses.shape[1]
    est_waves = bandpasses[:, Nwaves_per_bp//2]

    starting_itr = len(data['images'])
    if len(data['commands'])>0:
        total_command = copy.copy(data['commands'][-1])
    else:
        total_command = xp.zeros((M.Nact,M.Nact))
    
    del_command = xp.zeros((M.Nact,M.Nact))
    del_acts0 = np.zeros(M.Nacts)
    for i in range(Nitr):
        print('Running estimation algorithm ...')
        
        if pwp_params is not None: 
            E_abs = run_pwp_bb(I, M, ensure_np_array(total_command[M.dm_mask]), **pwp_params)
        else:
            E_abs = calc_wfs(I, est_waves, control_mask, plot=0)

        # print(E_abs.shape)
        # print(est_waves)
        print('Computing EFC command with L-BFGS')
        current_acts = ensure_np_array(total_command[M.dm_mask])
        res = minimize(val_and_grad_bb, 
                       jac=True, 
                       x0=del_acts0,
                       args=(M, current_acts, E_abs, control_mask, est_waves, reg_cond, False, False, False), 
                       method='L-BFGS-B',
                       tol=bfgs_tol,
                       options=bfgs_opts,
                       )

        del_acts = gain * res.x
        del_command[M.dm_mask] = del_acts
        total_command = (1-leakage)*total_command + del_command

        # I.add_dm(del_command)
        I.set_dm(total_command)

        I.return_ni = True
        I.subtract_dark = True
        I.waves = bandpasses.flatten()
        image_ni = I.snap()
        mean_ni = xp.mean(image_ni[control_mask])

        data['images'].append(copy.copy(image_ni))
        data['efields'].append(copy.copy(E_abs))
        data['commands'].append(copy.copy(total_command))
        data['del_commands'].append(copy.copy(del_command))
        data['bfgs_tols'].append(bfgs_tol)
        data['reg_conds'].append(reg_cond)
        
        imshow3(del_command, total_command, image_ni, 
                f'Iteration {starting_itr + i:d}: $\delta$DM', 
                'Total DM Command', 
                f'Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                vmin1=-xp.max(xp.abs(del_command)), vmax1=xp.max(xp.abs(del_command)),
                vmin2=-xp.max(xp.abs(total_command)), vmax2=xp.max(xp.abs(total_command)),
                pxscl3=I.psf_pixelscale_lamDc, lognorm3=True, vmin3=vmin)

    return data

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_data(data, vmin=1e-9, vmax=1e-4):
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

    fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(15,10), dpi=125, gridspec_kw={'width_ratios': [1, 1, 1.35], })
    ext = psf_pixelscale_lamD*npsf/2
    extent = [-ext, ext, -ext, ext]

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
    ax[1].set_position([0.212, 0.3, 0.25, 0.25])

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
    # ax[2].set_aspect(1.15)

    # plt.subplots_adjust(wspace=0.45)
    # fig.savefig('figs/iefc_bb_plots.pdf', format='pdf', bbox_inches="tight")

