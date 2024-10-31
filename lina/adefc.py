from .math_module import xp, _scipy, ensure_np_array
import lina.utils as utils
from lina.imshows import imshow1, imshow2, imshow3

import numpy as np
from scipy.optimize import minimize
import time
import copy

def run_pwp(I, 
            M, 
            current_acts, 
            control_mask, 
            probes, probe_amp, 
            reg_cond=1e-3, 
            plot=False,
            plot_est=False,
            ):
    
    Nmask = int(control_mask.sum())
    Nprobes = probes.shape[0]

    I.subtract_dark = False
    Ip = []
    In = []
    for i in range(Nprobes):
        for s in [-1, 1]:
            I.add_dm(s*probe_amp*probes[i])
            coro_im = I.snap()
            I.add_dm(-s*probe_amp*probes[i]) # remove probe from DM

            if s==-1: 
                In.append(coro_im)
            else: 
                Ip.append(coro_im)
        
    E_probes = xp.zeros((probes.shape[0], 2*Nmask))
    I_diff = xp.zeros((probes.shape[0], Nmask))
    for i in range(Nprobes):
        if i==0: 
            E_nom = M.forward(current_acts, use_vortex=True)
        E_with_probe = M.forward(xp.array(current_acts) + xp.array(probe_amp*probes[i])[M.dm_mask], use_vortex=True)
        E_probe = E_with_probe - E_nom
        diff_im = Ip[i] - In[i]
        if plot:
            imshow3(diff_im, xp.abs(E_probe), xp.angle(E_probe),
                    'Difference Image', f'Probe {i+1}: '+'$|E_{probe}|$', f'Probe {i+1}: '+r'$\angle E_{probe}$', 
                    cmap3='twilight')
            
        E_probes[i, ::2] = E_probe[control_mask].real
        E_probes[i, 1::2] = E_probe[control_mask].imag
        I_diff[i, :] = diff_im[control_mask]
    
    # Use batch process to estimate each pixel individually
    E_est = xp.zeros(Nmask, dtype=xp.complex128)
    for i in range(Nmask):
        delI = I_diff[:, i]
        H = 4*xp.array([E_probes[:,2*i], E_probes[:,2*i + 1]]).T
        Hinv = xp.linalg.pinv(H.T@H, reg_cond)@H.T
    
        est = Hinv.dot(delI)

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = xp.zeros((I.npsf,I.npsf), dtype=xp.complex128)
    E_est_2d[control_mask] = E_est

    if plot or plot_est:
        I_est = xp.abs(E_est_2d)**2
        P_est = xp.angle(E_est_2d)
        imshow2(I_est, P_est, 
                'Estimated Intensity', 'Estimated Phase',
                lognorm1=True, vmin1=xp.max(I_est)/1e3, 
                cmap2='twilight',
                pxscl=M.psf_pixelscale_lamD)
    return E_est_2d

def run(I, 
        M, 
        val_and_grad,
        control_mask,
        data,
        est_fun=None, est_params=None,
        Nitr=3, 
        reg_cond=1e-2,
        bfgs_tol=1e-3,
        bfgs_opts=None,
        gain=0.5, 
        leakage=0.0, 
        vmin=1e-9, 
        ):

    starting_itr = len(data['images'])
    if len(data['commands'])>0:
        total_command = copy.copy(data['commands'][-1])
    else:
        total_command = xp.zeros((M.Nact,M.Nact))

    del_command = xp.zeros((M.Nact,M.Nact))
    del_acts0 = np.zeros(M.Nacts)
    for i in range(Nitr):
        print('Running estimation algorithm ...')
        
        if est_fun is not None and est_params is not None: 
            E_ab = est_fun(I, M, ensure_np_array(total_command[M.dm_mask]), **est_params)
        else:
            E_ab = I.calc_wf()
        
        print('Computing EFC command with L-BFGS')
        res = minimize(val_and_grad, 
                       jac=True, 
                       x0=del_acts0,
                       args=(M, ensure_np_array(total_command[M.dm_mask]), E_ab, reg_cond, control_mask), 
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
        image_ni = I.snap()
        mean_ni = xp.mean(image_ni[control_mask])

        data['images'].append(copy.copy(image_ni))
        data['efields'].append(copy.copy(E_ab))
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
                pxscl3=I.psf_pixelscale_lamD, lognorm3=True, vmin3=vmin)

    
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

    fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(16,8), dpi=125, gridspec_kw={'width_ratios': [1, 1, 1.35], })
    ext = psf_pixelscale_lamD*npsf/2
    extent = [-ext, ext, -ext, ext]

    im1 = ax[0].imshow(ref_im, norm=LogNorm(vmax=vmax, vmin=vmin), cmap='magma', extent=extent)
    ax[0].set_title(f'Reference Image:\nMean Contrast = {mean_nis[0]:.3e}', fontsize=14)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im1, cax=cax)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)

    im2 = ax[1].imshow( best_im, norm=LogNorm(vmax=vmax, vmin=vmin), cmap='magma', extent=extent)
    ax[1].set_title(f'Best Iteration:\nMean Contrast = {mean_nis[ibest]:.3e}', fontsize=14)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im2, cax=cax,)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)

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
    ax[2].set_aspect(1.15)

    plt.subplots_adjust(wspace=0.45)
    # fig.savefig('figs/iefc_bb_plots.pdf', format='pdf', bbox_inches="tight")

