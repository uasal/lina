from wfsc_tests.math_module import xp
from wfsc_tests import utils

def run_pwp_broad(sysi, wavelengths, probes, dark_mask, 
                  use, jacobian=None, model=None, 
                  rcond=1e-15, 
                  use_noise=False, display=False):
    nmask = dark_mask.sum()
    nwaves = len(wavelengths)
    nprobes = probes.shape[0]
    
    dm_ref = sysi.get_dm()
    amps = np.linspace(-1, 1, 2) # for generating a negative and positive probe
    
    Ip = []
    In = []
    for i,probe in enumerate(probes):
        for amp in amps:
            sysi.add_dm(amp*probe)
            
            bb_im = 0
            for wavelength in wavelengths:
                sysi.wavelength = wavelength
                bb_im += sysi.snap()
                
            if amp==-1: 
                In.append(bb_im)
            else: 
                Ip.append(bb_im)
                
            sysi.add_dm(-amp*probe) # remove probe from DM
            
        if display:
            misc.myimshow3(Ip[i], In[i], Ip[i]-In[i],
                           'Probe {:d} Positive Image'.format(i+1), 'Probe {:d} Negative Image'.format(i+1),
                           'Intensity Difference',
                           lognorm1=True, lognorm2=True, vmin1=Ip[i].max()/1e6, vmin2=In[i].max()/1e6,
                          )
            
    E_probes = np.zeros((2*nmask*nwaves*nprobes,))
    I_diff = np.zeros((nprobes*nmask,))
    for i in range(nprobes):
        I_diff[i*nmask:(i+1)*nmask] = (Ip[i] - In[i])[dark_mask]
        
        for j in range(nwaves):
#             jac_wave = jacobian[j*2*nmask:(j+1)*2*nmask]
#             E_probe = jac_wave.dot(np.array(probes[i].flatten()))
            
            if (use=='jacobian' or use=='j') and jacobian is not None:
                jac_wave = jacobian[j*2*nmask:(j+1)*2*nmask]
                E_probe = jac_wave.dot(np.array(probes[i].flatten()))
            elif (use=='model' or use=='m') and model is not None:
                model.wavelength = wavelengths[j]
                E_full = model.calc_psf().wavefront.get()[dark_mask]

                model.add_dm(probes[i])
                E_full_probe = model.calc_psf().wavefront.get()[dark_mask]
                model.add_dm(-probes[i])

                E_probe = E_full_probe - E_full
                E_probe = np.concatenate((E_probe.real, E_probe.imag))
            
            E_probes[i*nwaves*2*nmask + j*2*nmask : i*nwaves*2*nmask + (j+1)*2*nmask] = E_probe

    B = np.diag(np.ones((nmask,2*nmask))[0], k=0)[:nmask,:2*nmask] \
        + np.diag(np.ones((nmask,2*nmask))[0], k=nmask)[:nmask,:2*nmask]
    Bfull = np.tile(B, nwaves )
#     misc.myimshow(B, figsize=(10,4))
    
    for i in range(nprobes):
        for j in range(nwaves):
            h = 4 * B @ np.diag( E_probes[ i*nwaves*2*nmask + j*2*nmask : i*nwaves*2*nmask + (j+1)*2*nmask ] )
            H_inv = h if j==0 else np.hstack((H_inv,h))
            
#             E_probe_2d = np.zeros((sysi.npsf,sysi.npsf), dtype=np.complex128)
#             np.place(E_probe_2d, mask=dark_mask, 
#                      vals=E_probes[i*nwaves*2*nmask + j*2*nmask : i*nwaves*2*nmask + (j+1)*2*nmask][:nmask] + \
#                           1j*E_probes[i*nwaves*2*nmask + j*2*nmask : i*nwaves*2*nmask + (j+1)*2*nmask][nmask:])
#             misc.myimshow2(np.abs(E_probe_2d), np.angle(E_probe_2d))
            
        Hinv = H_inv if i==0 else np.vstack((Hinv,H_inv))
        print('Hinv.shape', Hinv.shape)
    
    Hinv = cp.array(Hinv)
    H = (cp.linalg.pinv(Hinv.T@Hinv, rcond)@Hinv.T).get()
    
    E_est = H.dot(I_diff)
    
    Es_2d = []
    for j in range(nwaves):
        E = E_est[j*2*nmask:j*2*nmask+nmask] + 1j*E_est[j*2*nmask+nmask:j*2*nmask+2*nmask] 
        
        E_2d = np.zeros((sysi.npsf,sysi.npsf), dtype=np.complex128)
        np.place(E_2d, mask=dark_mask, vals=E)
        Es_2d.append(E_2d)
        
    Es_2d = np.array(Es_2d)
    
    return Es_2d


