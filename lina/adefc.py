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
            E_nom = M.forward(current_acts, use_vortex=True, use_wfe=True)
        E_with_probe = M.forward(xp.array(current_acts) + xp.array(probe_amp*probes[i])[M.dm_mask], use_vortex=True, use_wfe=True)
        E_probe = E_with_probe - E_nom

        if plot:
            imshow3(xp.abs(E_probe), xp.angle(E_probe), Ip[i]-In[i], 
                    f'Probe {i+1}: '+'$|E_{probe}|$', f'Probe {i+1}: '+r'$\angle E_{probe}$', 'Difference Image', 
                    cmap2='twilight')
            
        E_probes[i, ::2] = E_probe[control_mask].real
        E_probes[i, 1::2] = E_probe[control_mask].imag

        # I_diff[i:(i+1), :] = (Ip[i] - In[i])[control_mask]
        I_diff[i, :] = (Ip[i] - In[i])[control_mask]
    
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
                lognorm1=True, vmin1=xp.max(I_est)/1e4, 
                cmap2='twilight',
                pxscl=M.psf_pixelscale_lamD)
    return E_est_2d

def run(I, 
        M, 
        val_and_grad,
        control_mask,
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
        total_command = xp.zeros((M.Nact,M.Nact))
    del_command = xp.zeros((M.Nact,M.Nact))
    del_acts0 = np.zeros(M.Nacts)
    for i in range(Nitr):
        print('Running estimation algorithm ...')
        I.subtract_dark = False
        
        if est_fun is not None and est_params is not None: 
            E_ab = est_fun(I, M, ensure_np_array(total_command[M.dm_mask]), **est_params)
        else:
            E_ab = I.calc_wf()
        
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


