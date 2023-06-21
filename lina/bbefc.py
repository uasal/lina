from wfsc_tests.math_module import xp
from wfsc_tests import utils

def build_jacobian(sysi, wavelengths, epsilon, dark_mask, display=False):
    start = time.time()
    print('Building Jacobian.')
    
    responses = []
    amps = np.linspace(-epsilon, epsilon, 2) # for generating a negative and positive actuator poke
    
    dm_mask = sysi.dm_mask.flatten()
    
    num_modes = sysi.Nact**2
    modes = np.eye(num_modes) # each column in this matrix represents a vectorized DM shape where one actuator has been poked
    
    for wavelength in wavelengths:
        sysi.wavelength = wavelength
        print('Calculating sensitivity for wavelength {:.3e}'.format(wavelength))
        
        for i, mode in enumerate(modes):
            if dm_mask[i]==1:
                response = 0
                for amp in amps:
                    mode = mode.reshape(sysi.Nact,sysi.Nact)

                    sysi.add_dm(amp*mode)

                    psf = sysi.calc_psf()
                    wavefront = psf.wavefront
                    response += amp*wavefront/np.var(amps)

                    sysi.add_dm(-amp*mode)

                if display:
                    misc.myimshow2(cp.abs(response), cp.angle(response))

                response = response.flatten().get()[dark_mask.flatten()]

            else:
                response = np.zeros((sysi.npsf, sysi.npsf), dtype=np.complex128).flatten()[dark_mask.flatten()]

            responses.append(np.concatenate((response.real, response.imag)))

            print('\tCalculated response for mode {:d}/{:d}. Elapsed time={:.3f} sec.'.format(i+1,num_modes,time.time()-start))
        
    jacobian = np.array(responses).T
    
    for i in range(len(wavelengths)):
        jac_new = jac[:,:sysi.Nact**2] if i==0 else np.concatenate((jac_new, jac[:,i*sysi.Nact**2:(i+1)*sysi.Nact**2]), axis=0)
        
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    return jac_new

def run_efc_perfect(sysi, 
                    wavelengths, 
                    jac, 
                    reg_fun,
                    reg_conds,
                    dark_mask, 
                    Imax_unocc,
                    efc_loop_gain=0.5, 
                    iterations=5, 
                    display_all=False, 
                    display_current=True,
                    plot_sms=True):
    # This function is only for running EFC simulations
    print('Beginning closed-loop EFC simulation.')    
    commands = []
    images = []
    
    start = time.time()
    
    jac = cp.array(jac) if isinstance(jac, np.ndarray) else jac
    
    U, s, V = cp.linalg.svd(jac, full_matrices=False)
    alpha2 = cp.max( cp.diag( cp.real( jac.conj().T @ jac ) ) )
    print('Max singular value squared:\t', s.max()**2)
    print('alpha^2:\t\t\t', alpha2) 
    
    N_DH = dark_mask.sum()
    
    dm_ref = sysi.get_dm()
    dm_command = np.zeros((sysi.Nact, sysi.Nact)) 
    print()
    for i in range(iterations+1):
        print('\tRunning iteration {:d}/{:d}.'.format(i, iterations))
        
        if i==0 or i in reg_conds[0]:
            reg_cond_ind = np.argwhere(i==reg_conds[0])[0][0]
            reg_cond = reg_conds[1, reg_cond_ind]
            print('\tComputing EFC matrix via ' + reg_fun.__name__ + ' with condition value {:.2e}'.format(reg_cond))
            efc_matrix = reg_fun(jac, reg_cond).get()
        
        sysi.set_dm(dm_ref + dm_command) 
        
        psf_bb = 0
        electric_fields = [] # contains the e-field for each discrete wavelength
        for wavelength in wavelengths:
            sysi.wavelength = wavelength
            psf = sysi.calc_psf()
            electric_fields.append(psf.wavefront[dark_mask].get())
            psf_bb += psf.intensity.get()
            
        commands.append(sysi.get_dm())
        images.append(copy.copy(psf_bb))
        
        for j in range(len(wavelengths)):
            xnew = np.concatenate( (electric_fields[j].real, electric_fields[j].imag) )
            x = xnew if j==0 else np.concatenate( (x,xnew) )
        del_dm = efc_matrix.dot(x).reshape(sysi.Nact,sysi.Nact)
        
        dm_command -= efc_loop_gain * del_dm
        
        if display_current or display_all:
            if not display_all: clear_output(wait=True)
                
            fig,ax = misc.myimshow2(commands[i], images[i], 
                                        'DM', 'Image: Iteration {:d}'.format(i),
                                        lognorm2=True, vmin2=1e-12,
                                        return_fig=True, display_fig=True)
            if plot_sms:
                sms_fig = utils.sms(U, s, alpha2, cp.array(x), N_DH, Imax_unocc, i)
        
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return commands, images

def run_efc_pwp(sysi, 
                wavelengths,
                probes,
                    jac, 
                    reg_fun,
                    reg_conds,
                    dark_mask, 
                    Imax_unocc,
                    efc_loop_gain=0.5, 
                    iterations=5, 
                    display_all=False, 
                    display_current=True,
                    plot_sms=True):
    print('Beginning closed-loop EFC simulation.')
    
    commands = []
    efields = []
    images = []
    
    start=time.time()
    
    jac_cp = cp.array(jac) if isinstance(jac, np.ndarray) else jac
    
    U, s, V = cp.linalg.svd(jac_cp, full_matrices=False)
    alpha2 = cp.max( cp.diag( cp.real( jac_cp.conj().T @ jac_cp ) ) )
    print('Max singular value squared:\t', s.max()**2)
    print('alpha^2:\t\t\t', alpha2) 
    
    N_DH = dark_mask.sum()
    
    dm_ref = sysi.get_dm()
    dm_command = np.zeros((sysi.Nact, sysi.Nact)) 
    for i in range(iterations+1):
        print('\tRunning iteration {:d}/{:d}.'.format(i+1, iterations))
        
        if i==0 or i in reg_conds[0]:
            reg_cond_ind = np.argwhere(i==reg_conds[0])[0][0]
            reg_cond = reg_conds[1, reg_cond_ind]
            print('\tComputing EFC matrix via ' + reg_fun.__name__ + ' with condition value {:.2e}'.format(reg_cond))
            efc_matrix = reg_fun(jac_cp, reg_cond).get()
        
        sysi.set_dm(dm_ref + dm_command)
        E_ests = pwp.run_pwp_broad(sysi, wavelengths, probes, dark_mask, use='j', jacobian=jac/2)
        I_est = 0
        for j in range(len(wavelengths)):
            I_est += np.abs(E_ests[j])**2/len(wavelengths)
            
        if sysi.is_model:
            I_exact = 0
            for wavelength in wavelengths:
                sysi.wavelength = wavelength
                I_exact += sysi.snap()/len(wavelengths)
        else:
            I_exact = sysi.snap()
            
        for j in range(len(wavelengths)):
            xnew = np.concatenate( (E_ests[j][dark_mask].real, E_ests[j][dark_mask].imag) )
            x = xnew if j==0 else np.concatenate( (x,xnew) )
        del_dm = efc_matrix.dot(x).reshape(sysi.Nact,sysi.Nact)
        
        commands.append(sysi.get_dm())
        efields.append(copy.copy(E_ests))
        
        images.append(copy.copy(I_exact))
        
        dm_command -= efc_loop_gain * del_dm.reshape(sysi.Nact,sysi.Nact)
        
        if display_current or display_all:
            if not display_all: clear_output(wait=True)
                
            fig,ax = misc.myimshow3(commands[i], I_est, I_exact, 
                                        'DM', 'Estimated Intensity', 'Image: Iteration {:d}'.format(i),
                                        lognorm2=True, vmin2=1e-12, lognorm3=True, vmin3=1e-12,
                                        return_fig=True, display_fig=True)
            if plot_sms:
                sms_fig = utils.sms(U, s, alpha2, cp.array(x), N_DH, Imax_unocc, i)
        
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return commands, efields, images
