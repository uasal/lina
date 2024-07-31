from .math_module import xp, _scipy, ensure_np_array
from . import utils
from .imshows import imshow1, imshow2, imshow3

import numpy as np
import astropy.units as u
from astropy.io import fits
import poppy
import scipy
from scipy.optimize import minimize
import os
from pathlib import Path
import time
import copy

def create_poke_modes(I):
    poke_modes = xp.zeros((I.Nacts, I.Nact, I.Nact))
    count = 0
    for i in range(I.Nact):
        for j in range(I.Nact):
            if I.dm_mask[i,j]:
                poke_modes[count, i,j] = 1
                count += 1
    
    return poke_modes

def compute_jacobian(M,
                     modes,
                     control_mask,
                     amp=1e-9):
    Nmodes = modes.shape[0]
    jac = xp.zeros((2*m.Nmask, Nmodes))
    for i in range(Nmodes):
        E_pos = M.forward(amp*modes[i][M.dm_mask], use_wfe=True, use_vortex=True)[control_mask]
        E_neg = M.forward(-amp*modes[i][M.dm_mask], use_wfe=True, use_vortex=True)[control_mask]
        response = (E_pos - E_neg)/(2*amp)
        jac[::2,i] = xp.real(response)
        jac[1::2,i] = xp.imag(response)

    return jac

def sim_efc(M,
            control_matrix,  
            control_mask,
            Nitr=3, 
            gain=0.5, 
            all_ims=[], 
            all_efs=[],
            all_commands=[],
            ):
    
    starting_itr = len(all_ims)

    if len(all_commands)>0:
        total_command = copy.copy(all_commands[-1])
    else:
        total_command = xp.zeros((M.Nact,M.Nact))
    del_command = xp.zeros((M.Nact,M.Nact))
    E_ab = xp.zeros(2*M.Nmask)
    for i in range(Nitr):
        E_est = M.forward(total_command[m.dm_mask], use_vortex=True, use_wfe=True,)
        E_ab[::2] = E_est.real[control_mask]
        E_ab[1::2] = E_est.imag[control_mask]

        del_acts = -gain * control_matrix.dot(E_ab)
        del_command[M.dm_mask] = del_acts
        total_command += del_command
        
        image_ni = M.snap(total_command[M.dm_mask], use_vortex=True, use_wfe=True)

        mean_ni = xp.mean(image_ni[control_mask])

        all_ims.append(copy.copy(image_ni))
        all_efs.append(copy.copy(E_ab))
        all_commands.append(copy.copy(total_command))

        imshow3(del_command, total_command, image_ni, 
                f'Iteration {starting_itr + i + 1:d}: $\delta$DM', 
                'Total DM Command', 
                f'Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=M.psf_pixelscale_lamD, lognorm3=True, vmin3=1e-9)

    return all_ims, all_efs, all_commands

def sim_pwp(m, current_acts, control_mask,
            probes, probe_amp, 
            reg_cond=1e-3, 
            plot=False,
            plot_est=False):
    
    Nmask = int(control_mask.sum())
    Nprobes = probes.shape[0]

    Ip = []
    In = []
    for i in range(Nprobes):
        for s in [-1, 1]:
            coro_im = m.snap(current_acts + s*probe_amp*probes[i][m.dm_mask], use_vortex=True, use_wfe=True)

            if s==-1: 
                In.append(coro_im)
            else: 
                Ip.append(coro_im)
            
        if plot:
            imshow3(Ip[i], In[i], Ip[i]-In[i], lognorm1=True, lognorm2=True, pxscl=m.psf_pixelscale_lamD)
            
    E_probes = xp.zeros((probes.shape[0], 2*Nmask))
    I_diff = xp.zeros((probes.shape[0], Nmask))
    for i in range(Nprobes):
        if i==0: 
            E_nom = m.forward(current_acts, use_vortex=True, use_wfe=True)
        E_with_probe = m.forward(current_acts + probe_amp*probes[i][m.dm_mask], use_vortex=True, use_wfe=True)
        E_probe = E_with_probe - E_nom

        if plot:
            imshow2(xp.abs(E_probe), xp.angle(E_probe),
                    f'Probe {i+1}: '+'$|E_{probe}|$', f'Probe {i+1}: '+r'$\angle E_{probe}$', 
                    cmap2='twilight')
            
        E_probes[i, ::2] = E_probe[control_mask].real
        E_probes[i, 1::2] = E_probe[control_mask].imag
        I_diff[i, :] = (Ip[i] - In[i])[control_mask]
    
    # Use batch process to estimate each pixel individually
    E_est = xp.zeros(Nmask, dtype=xp.complex128)
    for i in range(Nmask):
        delI = I_diff[:, i]
        M = 4*xp.array([E_probes[:,2*i], E_probes[:,2*i + 1]]).T
        Minv = xp.linalg.pinv(M.T@M, reg_cond)@M.T
    
        est = Minv.dot(delI)

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = xp.zeros((m.npsf,m.npsf), dtype=xp.complex128)
    E_est_2d[control_mask] = E_est
    
    if plot or plot_est:
        I_est = xp.abs(E_est_2d)**2
        P_est = xp.angle(E_est_2d)
        imshow2(I_est, P_est, 
                'Estimated Intensity', 'Estimated Phase',
                lognorm1=True, vmin1=xp.max(I_est)/1e4, 
                cmap2='twilight',
                pxscl=m.psf_pixelscale_lamD)
    return E_est_2d

def run_pwp(sysi, m, current_acts, 
            control_mask, 
            probes, probe_amp, 
            reg_cond=1e-3, 
            plot=False,
            plot_est=False):
    
    Nmask = int(control_mask.sum())
    Nprobes = probes.shape[0]

    Ip = []
    In = []
    for i in range(Nprobes):
        for s in [-1, 1]:
            sysi.add_dm(s*probe_amp*probes[i])
            coro_im = sysi.snap()
            sysi.add_dm(-s*probe_amp*probes[i]) # remove probe from DM

            if s==-1: 
                In.append(coro_im)
            else: 
                Ip.append(coro_im)
            
        if plot:
            imshow3(Ip[i], In[i], Ip[i]-In[i], lognorm1=True, lognorm2=True, pxscl=sysi.psf_pixelscale_lamD)
            
    E_probes = xp.zeros((probes.shape[0], 2*Nmask))
    I_diff = xp.zeros((probes.shape[0], Nmask))
    for i in range(Nprobes):
        if i==0: 
            E_nom = m.forward(current_acts, use_vortex=True, use_wfe=True)
        E_with_probe = m.forward(xp.array(current_acts) + xp.array(probe_amp*probes[i])[m.dm_mask], use_vortex=True, use_wfe=True)
        E_probe = E_with_probe - E_nom

        if plot:
            imshow2(xp.abs(E_probe), xp.angle(E_probe),
                    f'Probe {i+1}: '+'$|E_{probe}|$', f'Probe {i+1}: '+r'$\angle E_{probe}$',
                    cmap2='viridis')
            
        E_probes[i, ::2] = E_probe[control_mask].real
        E_probes[i, 1::2] = E_probe[control_mask].imag

        # I_diff[i:(i+1), :] = (Ip[i] - In[i])[control_mask]
        I_diff[i, :] = (Ip[i] - In[i])[control_mask]
    
    # Use batch process to estimate each pixel individually
    E_est = xp.zeros(Nmask, dtype=xp.complex128)
    for i in range(Nmask):
        delI = I_diff[:, i]
        M = 4*xp.array([E_probes[:,2*i], E_probes[:,2*i + 1]]).T
        Minv = xp.linalg.pinv(M.T@M, reg_cond)@M.T
    
        est = Minv.dot(delI)

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = xp.zeros((sysi.npsf,sysi.npsf), dtype=xp.complex128)
    E_est_2d[control_mask] = E_est

    if plot or plot_est:
        I_est = xp.abs(E_est_2d)**2
        P_est = xp.angle(E_est_2d)
        imshow2(I_est, P_est, 
                'Estimated Intensity', 'Estimated Phase',
                lognorm1=True, vmin1=xp.max(I_est)/1e4, 
                cmap2='twilight',
                pxscl=m.psf_pixelscale_lamD)
    return E_est_2d


def sim(m, val_and_grad, control_mask,
        est_fun=None, est_params=None,
        Nitr=3, 
        reg_cond=1e-2,
        bfgs_tol=1e-3,
        bfgs_opts=None,
        gain=0.5, 
        all_ims=[], 
        all_efs=[],
        all_commands=[],
        ):
    starting_itr = len(all_ims)

    if len(all_commands)>0:
        total_command = copy.copy(all_commands[-1])
    else:
        total_command = xp.zeros((m.Nact,m.Nact))
    del_command = xp.zeros((m.Nact,m.Nact))
    del_acts0 = np.zeros(m.Nacts)
    for i in range(Nitr):
        if est_fun is None: 
            E_ab = m.forward(total_command[m.dm_mask], use_vortex=True, use_wfe=True,)
        else: 
            E_ab = est_fun(m, total_command[m.dm_mask], **est_params)

        res = minimize(val_and_grad, 
                       jac=True, 
                       x0=del_acts0,
                       args=(m, ensure_np_array(total_command[m.dm_mask]), E_ab, reg_cond, control_mask), 
                       method='L-BFGS-B',
                       tol=bfgs_tol,
                       options=bfgs_opts,
                       )

        del_acts = gain * res.x
        del_command[m.dm_mask] = del_acts
        total_command += del_command

        # image_ni = xp.abs(m.forward(total_command[m.dm_mask], use_vortex=True, use_wfe=True))**2
        image_ni = m.snap(total_command[m.dm_mask])
        mean_ni = xp.mean(image_ni[control_mask])

        all_ims.append(copy.copy(image_ni))
        all_efs.append(copy.copy(E_ab))
        all_commands.append(copy.copy(total_command))

        imshow3(del_command, total_command, image_ni, 
                f'Iteration {starting_itr + i + 1:d}: $\delta$DM', 
                'Total DM Command', 
                f'Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=m.psf_pixelscale_lamD, lognorm3=True, vmin3=1e-10)

    return all_ims, all_efs, all_commands

def run(I, M, 
        val_and_grad,
        control_mask,
        est_fun, est_params,
        Nitr=3, 
        reg_cond=1e-2,
        bfgs_tol=1e-3,
        bfgs_opts=None,
        gain=0.5, 
        all_ims=[], 
        all_efs=[],
        all_commands=[],
        ):
    
    starting_itr = len(all_ims)

    if len(all_commands)>0:
        total_command = copy.copy(all_commands[-1])
    else:
        total_command = xp.zeros((M.Nact,m.Nact))
    del_command = xp.zeros((M.Nact,M.Nact))
    del_acts0 = np.zeros(M.Nacts)
    for i in range(Nitr):
        print('Running estimation algorithm ...')
        I.subtract_dark = False
        E_ab = est_fun(I, M, ensure_np_array(total_command[M.dm_mask]), **est_params)
        
        print('Computing EFC command with L-BFGS')
        res = minimize(val_and_grad, 
                       jac=True, 
                       x0=del_acts0,
                    #    args=(m, E_ab, reg_cond, E_target, E_model_nom), 
                       args=(M, ensure_np_array(total_command[M.dm_mask]), E_ab, reg_cond, control_mask), 
                       method='L-BFGS-B',
                       tol=bfgs_tol,
                       options=bfgs_opts,
                       )

        del_acts = gain * res.x
        del_command[M.dm_mask] = del_acts
        total_command += del_command

        I.add_dm(del_command)
        I.subtract_dark = True
        image_ni = I.snap()
        mean_ni = xp.mean(image_ni[control_mask])

        all_ims.append(copy.copy(image_ni))
        all_efs.append(copy.copy(E_ab))
        all_commands.append(copy.copy(total_command))
        
        imshow3(del_command, total_command, image_ni, 
                f'Iteration {starting_itr + i + 1:d}: $\delta$DM', 
                'Total DM Command', 
                f'Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                vmin1=-xp.max(xp.abs(del_command)), vmax1=xp.max(xp.abs(del_command)),
                vmin2=-xp.max(xp.abs(total_command)), vmax2=xp.max(xp.abs(total_command)),
                pxscl3=I.psf_pixelscale_lamD, lognorm3=True, vmin3=1e-10)

    return all_ims, all_efs, all_commands


