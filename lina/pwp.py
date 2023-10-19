from .math_module import xp, _scipy, ensure_np_array
from . import utils
from . import imshows

import numpy as np
import astropy.units as u
from astropy.io import fits
from pathlib import Path
from importlib import reload
from IPython.display import clear_output
import time
import copy


def run_pwp_bp(sysi, 
               dark_mask, 
               probes,
               use='J', jacobian=None, model=None, 
               plot=False,
               plot_est=False):
    """_summary_

    Parameters
    ----------
    sysi : _type_
        _description_
    dark_mask : _type_
        _description_
    probes : _type_
        _description_
    use : str, optional
        _description_, by default 'J'
    jacobian : _type_, optional
        _description_, by default None
    model : _type_, optional
        _description_, by default None
    plot : bool, optional
        _description_, by default False
    plot_est : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    Nmask = int(dark_mask.sum())
    
    dm_ref = sysi.get_dm()
    amps = np.linspace(-1, 1, 2) # for generating a negative and positive probe
    
    Ip = []
    In = []
    for i,probe in enumerate(probes):
        for amp in amps:
            sysi.add_dm(amp*probe)
            psf = sysi.snap()
                
            if amp==-1: 
                In.append(psf)
            else: 
                Ip.append(psf)
                
            sysi.add_dm(-amp*probe) # remove probe from DM
            
        if plot:
            imshows.imshow3(Ip[i], In[i], Ip[i]-In[i], lognorm1=True, lognorm2=True, pxscl=sysi.psf_pixelscale_lamD)
            
    E_probes = xp.zeros((probes.shape[0], 2*Nmask))
    I_diff = xp.zeros((probes.shape[0], Nmask))
    for i in range(len(probes)):
        if (use=='jacobian' or use.lower()=='j') and jacobian is not None:
            E_probe = jacobian.dot(xp.array(probes[i][sysi.dm_mask.astype(bool)]))
            E_probe = E_probe[::2] + 1j*E_probe[1::2]
        elif (use=='model' or use=='m') and model is not None:
            if i==0: 
                E_full = model.calc_psf()[dark_mask]
                
            model.add_dm(probes[i])
            E_full_probe = model.calc_psf()[dark_mask]
            model.add_dm(-probes[i])
            
            E_probe = E_full_probe - E_full
            # print(type(E_probe))
            
        if plot:
            E_probe_2d = xp.zeros((sysi.npsf,sysi.npsf), dtype=xp.complex128)
            # print(xp)
            # print(type(E_probe_2d), type(dark_mask))
            xp.place(E_probe_2d, mask=dark_mask, vals=E_probe)
            imshows.imshow2(xp.abs(E_probe_2d), xp.angle(E_probe_2d),
                            f'Probe {i+1}: '+'$|E_{probe}|$', f'Probe {i+1}: '+r'$\angle E_{probe}$')
            
        E_probes[i, ::2] = E_probe.real
        E_probes[i, 1::2] = E_probe.imag

        I_diff[i:(i+1), :] = (Ip[i] - In[i])[dark_mask]
    
    # Use batch process to estimate each pixel individually
    E_est = xp.zeros(Nmask, dtype=xp.complex128)
    for i in range(Nmask):
        delI = I_diff[:, i]
        M = 4*xp.array([E_probes[:,2*i], E_probes[:,2*i + 1]]).T
        Minv = xp.linalg.pinv(M.T@M, 1e-2)@M.T
    
        est = Minv.dot(delI)

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = xp.zeros((sysi.npsf,sysi.npsf), dtype=xp.complex128)
    xp.place(E_est_2d, mask=dark_mask, vals=E_est)
    
    if plot or plot_est:
        imshows.imshow2(xp.abs(E_est_2d)**2, xp.angle(E_est_2d), 
                        'Estimated Intensity', 'Estimated Phase',
                        lognorm1=True, pxscl=sysi.psf_pixelscale_lamD)
    return E_est_2d

def run_pwp_redmond(sysi, dark_mask, 
                probes,
                use, jacobian=None, model=None, 
                rcond=1e-15,
                use_noise=False, display=False):
    nmask = dark_mask.sum()
    nprobes = probes.shape[0]
    
    dm_ref = sysi.get_dm()
    amps = np.linspace(-1, 1, 2) # for generating a negative and positive probe
    
    Ip = []
    In = []
    for i,probe in enumerate(probes):
        for amp in amps:
            sysi.add_dm(amp*probe)
            
            im = sysi.snap()
                
            if amp==-1: 
                In.append(im)
            else: 
                Ip.append(im)
                
            sysi.add_dm(-amp*probe) # remove probe from DM
            
        if display:
            misc.myimshow3(Ip[i], In[i], Ip[i]-In[i],
                           'Probe {:d} Positive Image'.format(i+1), 'Probe {:d} Negative Image'.format(i+1),
                           'Intensity Difference',
                           lognorm1=True, lognorm2=True, vmin1=Ip[i].max()/1e6, vmin2=In[i].max()/1e6,
                          )

    E_probes = np.zeros((2*nmask*nprobes,))
    I_diff = np.zeros((nmask*nprobes,))
    for i in range(nprobes):
        I_diff[ i*nmask : (i+1)*nmask ] = (Ip[i] - In[i])[dark_mask]

        if (use=='jacobian' or use=='j') and jacobian is not None:
            E_probe = jacobian.dot(np.array(probes[i].flatten())) # Use jacobian to model probe E-field at the focal plane
        elif (use=='model' or use=='m') and model is not None:
            if i==0: E_full = model.calc_psf().wavefront.get()[dark_mask]
                
            model.add_dm(probes[i])
            E_full_probe = model.calc_psf().wavefront.get()[dark_mask]
            model.add_dm(-probes[i])
            
            E_probe = E_full_probe - E_full
            E_probe = np.concatenate((E_probe.real, E_probe.imag))
            
        E_probes[ i*2*nmask : (i+1)*2*nmask ] = E_probe
        
        E_probe_2d = np.zeros((sysi.npsf,sysi.npsf), dtype=np.complex128)
        np.place(E_probe_2d, mask=dark_mask, 
                 vals=E_probes[i*2*nmask : (i+1)*2*nmask ][:nmask] + 1j*E_probes[i*2*nmask : (i+1)*2*nmask ][nmask:])
        if display:
            misc.myimshow2(np.abs(E_probe_2d), np.angle(E_probe_2d), 'E_probe Amp', 'E_probe Phase')
        
    B = np.diag(np.ones((nmask,2*nmask))[0], k=0)[:nmask,:2*nmask] + np.diag(np.ones((nmask,2*nmask))[0], k=nmask)[:nmask,:2*nmask]
    misc.myimshow(B, figsize=(10,4))
    print('B.shape', B.shape)
    
    for i in range(nprobes):
        h = 4 * B @ np.diag( E_probes[ i*2*nmask : (i+1)*2*nmask ] )
        Hinv = h if i==0 else np.vstack((Hinv,h))
    
    print('Hinv.shape', Hinv.shape)
    
    H = np.linalg.pinv(Hinv.T@Hinv, rcond)@Hinv.T
    print('H.shape', H.shape)
    
    E_est = H.dot(I_diff)
        
    E_est_2d = np.zeros((sysi.npsf,sysi.npsf), dtype=np.complex128)
    np.place(E_est_2d, mask=dark_mask, vals=E_est)
    
    return E_est_2d

def run_pwp_2011(sysi, dark_mask, 
                probes,
                use, jacobian=None, model=None, 
                rcond=1e-15,
                use_noise=False, display=False):
    nmask = dark_mask.sum()
    nprobes = probes.shape[0]
    
    I0 = sysi.snap()
    
    amps = np.linspace(-1, 1, 2) # for generating a negative and positive probe
    
    Ip = []
    In = []
    for i,probe in enumerate(probes):
        for amp in amps:
            sysi.add_dm(amp*probe)
            
            im = sysi.snap()
                
            if amp==-1: 
                In.append(im)
            else: 
                Ip.append(im)
                
            sysi.add_dm(-amp*probe) # remove probe from DM
            
        if display:
            misc.myimshow3(Ip[i], In[i], Ip[i]-In[i],
                           'Probe {:d} Positive Image'.format(i+1), 'Probe {:d} Negative Image'.format(i+1),
                           'Intensity Difference',
                           lognorm1=True, lognorm2=True, vmin1=Ip[i].max()/1e6, vmin2=In[i].max()/1e6,
                          )
    delI = np.zeros((nprobes*nmask,))
    delp = np.zeros((nprobes*nmask,), dtype=np.complex128) 
    for i in range(nprobes):
        delI[i*nmask:(i+1)*nmask] = (Ip[i]-In[i])[dark_mask]/2
        
        delp_amp = np.sqrt( (Ip[i]+In[i])[dark_mask]/2 - I0[dark_mask] )
        delp_amp[np.isnan(delp_amp)] = 0 # set the bad pixels to 0

        if (use=='jacobian' or use=='j') and jacobian is not None:
            del_p = jacobian.dot(np.array(probes[i].flatten())) 
            del_p = del_p[:nmask] + 1j*del_p[nmask:]
            
        delp_phs = np.angle(del_p)
        
        delp[i*nmask:(i+1)*nmask] = delp_amp * np.exp(1j*delp_phs)
        
    E_est = np.zeros((nmask,), dtype=cp.complex128)
    for i in range(nmask):
        
        M = 2*np.array([[-delp[i].imag, delp[i].real],
                        [-delp[i+nmask].imag, delp[i+nmask].real]])
        Minv = np.linalg.pinv(M.T@M, 1e-5)@M.T
    
        est = Minv.dot( [delI[i], delI[i+nmask]])

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = np.zeros((sysi.npsf,sysi.npsf), dtype=np.complex128)
    np.place(E_est_2d, mask=dark_mask, vals=E_est)
        
    return E_est_2d


