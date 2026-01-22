from .math_module import xp, xcipy, ensure_np_array
from lina import utils, props, dm

import numpy as np
import astropy.units as u
from astropy.io import fits
import os
from pathlib import Path
import time
import copy

import poppy
from scipy.signal import windows

import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize, CenteredNorm
from matplotlib.patches import Circle, Rectangle
from IPython.display import display, clear_output

def acts_to_command(acts, dm_mask):
    Nact = dm_mask.shape[0]
    command = xp.zeros((Nact,Nact))
    command[dm_mask] = xp.array(acts)
    return command

class MODEL():
    def __init__(
            self,
            wavelength_c=630e-9,
            wavelength=None, 
            npix=500,
            Ndef=502,
            N_vortex_lres=2048,
            vortex_win_diam=30, # diameter of the Tukey window in lambda/D to apply for the vortex model 
            vortex_hres_sampling=0.025, # lam/D per pixel; this value is chosen empirically
            vortex_dot_mask_diam_lamDc=0.5,
            dm_beam_diam=9.3e-3,
            lyot_pupil_diam=9.1e-3,
            lyot_stop_diam=8.6e-3,
            exit_pupil_prop_dist=None, 
            camsci_pxscl_lamDc=0.2,
            ncamsci=256,
            Nact=34,
            act_spacing=300e-6,
            act_coupling=0.15,
            PREFPM_AMP=None,
            PREFPM_OPD=None,
        ):

        self.wavelength_c = wavelength_c
        self.wavelength = wavelength_c if wavelength is None else wavelength
        # self.fsm_beam_diam = fsm_beam_diam
        self.dm_beam_diam = dm_beam_diam # as measured in the Fresnel model
        self.lyot_pupil_diam = lyot_pupil_diam
        self.lyot_stop_diam = lyot_stop_diam
        self.lyot_ratio = self.lyot_stop_diam/self.lyot_pupil_diam
        self.exit_pupil_prop_dist = exit_pupil_prop_dist
        self.camsci_pxscl_lamDc = camsci_pxscl_lamDc
        self.camsci_pxscl_lamD = self.camsci_pxscl_lamDc * self.wavelength_c/self.wavelength

        self.vortex_dot_mask_diam_lamDc = vortex_dot_mask_diam_lamDc
        self.vortex_dot_mask_diam_lamD = self.vortex_dot_mask_diam_lamDc * self.wavelength_c/self.wavelength
        
        self.tt_pv_to_rms = 1/4
        self.as_per_radian = 206264.806
        self.as_per_lamDc = ((self.wavelength_c / (self.dm_beam_diam*self.lyot_ratio) )*u.radian).to(u.arcsec)
        self.as_per_lamD = ((self.wavelength / (self.dm_beam_diam*self.lyot_ratio) )*u.radian).to(u.arcsec)
        
        self.npix = npix
        self.Ndef = Ndef
        self.def_oversample = self.Ndef / self.npix
        self.ncamsci = ncamsci

        self.Imax_ref = 1.0

        self.exit_pupil_pxscl = self.lyot_pupil_diam / self.npix

        ### INITIALIZE APERTURES ###
        pwf = poppy.FresnelWavefront(beam_radius=self.lyot_pupil_diam/2 * u.m, npix=self.npix, oversample=self.def_oversample)
        self.APERTURE = poppy.CircularAperture(radius=self.lyot_pupil_diam/2 * u.m).get_transmission(pwf)
        self.LYOTSTOP = poppy.CircularAperture(radius=self.lyot_ratio * self.lyot_pupil_diam/2 * u.m).get_transmission(pwf)
        self.BAP_MASK = self.APERTURE > 0.0

        self.PREFPM_AMP = PREFPM_AMP if PREFPM_AMP is not None else xp.ones_like(self.APERTURE)
        self.PREFPM_OPD = PREFPM_OPD if PREFPM_OPD is not None else xp.zeros_like(self.APERTURE)

        self.flip_dm = False
        self.reverse_lyot = False
        self.flip_lyot = False

        ### INITIALIZE DM PARAMETERS ###
        self.Nact = Nact
        self.dm_shape = (self.Nact, self.Nact)
        self.act_spacing = act_spacing
        self.dm_pxscl = self.dm_beam_diam / self.npix
        self.inf_sampling = self.act_spacing / self.dm_pxscl

        # construct DM influence function
        self.inf_fun = dm.make_gaussian_inf_fun(
            act_spacing=self.act_spacing, 
            sampling=self.inf_sampling, 
            coupling=act_coupling, 
            Nact=self.Nact+2,
        )
        self.Nsurf = self.inf_fun.shape[0]

        # construct DM mask
        y,x = (xp.indices((self.Nact, self.Nact)) - self.Nact//2 + 1/2)
        r = xp.sqrt(x**2 + y**2)
        self.dm_mask = r<(self.Nact/2 + 1/2)
        self.Nacts = int(self.dm_mask.sum())

        # construct DM modeling arrays and matrices
        self.inf_fun_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(self.inf_fun,)))

        xc = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2) # DM command coordinates
        yc = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)

        fx = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf)) # Influence function frequncy sampling
        fy = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))

        self.Mx_dm = xp.exp(-1j*2*np.pi*xp.outer(fx,xc)) # forward DM model MFT matrices
        self.My_dm = xp.exp(-1j*2*np.pi*xp.outer(yc,fy))
        self.Mx_dm_back = xp.exp(1j*2*np.pi*xp.outer(xc,fx)) # adjoint DM model MFT matrices
        self.My_dm_back = xp.exp(1j*2*np.pi*xp.outer(fy,yc))

         ### INITIALIZE VORTEX PARAMETERS ###
        self.N_vortex_lres = N_vortex_lres
        self.vortex_win_diam = vortex_win_diam # diameter of the Tukey window in lambda/D to apply for the vortex model 
        self.hres_sampling = vortex_hres_sampling # lam/D per pixel; this value is chosen empirically

        self.oversample_vortex = self.N_vortex_lres/self.npix
        self.lres_sampling = 1/self.oversample_vortex # low resolution sampling in lam/D per pixel
        self.lres_win_size = int(self.vortex_win_diam/self.lres_sampling)
        w1d = xp.array(windows.tukey(self.lres_win_size, 1, False))
        self.lres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_lres)
        self.vortex_lres = props.make_vortex_phase_mask(self.N_vortex_lres)

        self.N_vortex_hres = int(np.round(self.vortex_win_diam/self.hres_sampling))
        self.hres_win_size = int(self.vortex_win_diam/self.hres_sampling)
        w1d = xp.array(windows.tukey(self.hres_win_size, 1, False))
        self.hres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_hres)
        self.vortex_hres = props.make_vortex_phase_mask(self.N_vortex_hres)
        
        # Make the dot mask that we apply to the high resolution version of the vortex 
        # because that is the only branch that can see the spot well enough. I am using POPPY to 
        # do this because I am too lazy to make my own code that computes gray pixels for the spot.
        pwf = poppy.FresnelWavefront(beam_radius=self.N_vortex_hres/2 * self.hres_sampling * u.mm, npix=self.N_vortex_hres, oversample=1)
        self.hres_dot_mask = 1.0 - poppy.CircularAperture(radius= (self.vortex_dot_mask_diam_lamD/2) * u.mm).get_transmission(pwf)
        # utils.imshow([self.hres_dot_mask], npix=[20])

        self.windowed_vortex_lres = self.vortex_lres * (1 - self.lres_window) # apply low res (windowed) FPM
        self.windowed_vortex_hres = self.vortex_hres * self.hres_window * self.hres_dot_mask

        self.camsci_rotation = 0.0
        self.camsci_shift = np.array([0, 0])

        self.dm_commands = xp.zeros((10, self.Nact, self.Nact))
    
    def forward(
            self, 
            actuators, 
            wavelength=None, 
            use_vortex=True, 
            return_ints=False, 
            plot=False,
        ):

        if wavelength is None: wavelength = self.wavelength_c

        dm_command = xp.zeros((self.Nact,self.Nact))
        dm_command[self.dm_mask] = xp.array(actuators)
        mft_command = self.Mx_dm @ dm_command @ self.My_dm
        fourier_surf = self.inf_fun_fft * mft_command
        dm_surf = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(fourier_surf,))).real
        DM_PHASOR = xp.exp(1j * 4*xp.pi/wavelength * dm_surf)
        # if self.flip_dm: DM_PHASOR = xp.rot90(xp.rot90(DM_PHASOR))

        # Initialize the wavefront
        WFE =  self.PREFPM_AMP * xp.exp(1j * 2*xp.pi/wavelength * self.PREFPM_OPD)

        E_EP = self.APERTURE.astype(xp.complex128) * WFE / xp.sqrt(self.Imax_ref)

        E_DM = E_EP * utils.pad_or_crop(DM_PHASOR, self.Ndef)

        if use_vortex:
            E_DM_lres = utils.pad_or_crop(E_DM, self.N_vortex_lres) # pad to the larger array for the low res propagation
            E_FPM_lres = props.fft(E_DM_lres)
            E_FPM_lres *= self.windowed_vortex_lres
            E_LP_lres = props.ifft(E_FPM_lres)
            E_LP_lres = utils.pad_or_crop(E_LP_lres, self.Ndef) # crop to the desired wavefront dimension

            E_FPM_hres = props.mft_forward(
                E_DM, 
                self.npix, 
                self.N_vortex_hres, 
                self.hres_sampling, 
                convention='-', 
                pp_centering='odd', 
                fp_centering='odd', 
            )
            E_FPM_hres *= self.windowed_vortex_hres
            E_LP_hres = props.mft_reverse(
                E_FPM_hres,
                self.hres_sampling, 
                self.npix, 
                self.Ndef, 
                convention='+', 
                pp_centering='odd', 
                fp_centering='odd', 
            )

            E_LP = (E_LP_lres + E_LP_hres)
        else:
            E_LP = copy.copy(E_DM)

        # if self.reverse_lyot: E_LP = xp.rot90(xp.rot90(E_LP))
        # if self.flip_lyot: E_LP = xp.fliplr(E_LP)

        E_LS = self.LYOTSTOP.astype(xp.complex128) * E_LP

        if self.exit_pupil_prop_dist is not None:
            E_LS = utils.pad_or_crop(E_LS, 2*self.Ndef)
            E_FFFP = props.ang_spec(E_LS, wavelength, self.exit_pupil_prop_dist, self.exit_pupil_pxscl) # Final Front Focal Point
            # E_FFFP = utils.pad_or_crop(E_FFFP, self.Ndef)
        else:
            E_FFFP = copy.copy(E_LS)

        camsci_pxscl_lamD = self.camsci_pxscl_lamDc * self.wavelength_c/wavelength
        E_FP = props.mft_forward(E_FFFP, self.npix * self.lyot_ratio, self.ncamsci, camsci_pxscl_lamD)
        E_FP = xcipy.ndimage.rotate(E_FP, self.camsci_rotation, reshape=False, order=3)

        if use_vortex and plot: 
            utils.imshow(
                [xp.abs(E_EP), xp.angle(E_EP),
                 xp.abs(E_DM), xp.angle(E_DM),
                 xp.abs(E_LP_lres), xp.angle(E_LP_lres),
                 xp.abs(E_LP_hres), xp.angle(E_LP_hres),
                 xp.abs(E_LP), xp.angle(E_LP),
                 xp.abs(E_LS), xp.angle(E_LS),
                 xp.abs(E_FFFP), xp.angle(E_FFFP),
                 xp.abs(E_FP)**2, xp.angle(E_FP),], 
                titles=[
                    'EP Amplitude', 'EP Phase',
                ],
                # npix=8*[int(1.2*self.npix)], 
                cmaps=8*['plasma', 'twilight'],
                norms=7*[None,None] + [LogNorm(xp.max(xp.abs(E_FP)**2)/1e6), None],
                Nrows=8, Ncols=2,
                figsize=(12,42),
                hspace=0.2, wspace=0.2, 
            )

        if return_ints:
            return E_FP, E_EP, DM_PHASOR
        else:
            return E_FP

    def calc_wf(self):
        dm_command = xp.sum(self.dm_commands, axis=0)
        E_FP = self.forward(dm_command[self.dm_mask], self.wavelength, self.use_vortex)
        return E_FP

    def snap(self):
        dm_command = xp.sum(self.dm_commands, axis=0)
        E_FP = self.forward(dm_command[self.dm_mask], self.wavelength, self.use_vortex)
        im = xp.abs(E_FP)**2
        return im



def val_and_grad(
        del_acts, 
        M, 
        rmad_vars, 
        verbose=False, 
        plot=False, 
    ):
    # Convert array arguments into correct types
    del_acts = xp.array(del_acts)
    del_acts_waves = del_acts/M.wavelength_c

    current_acts = xp.array(rmad_vars['current_acts'])
    E_ab = xp.array(rmad_vars['E_ab'])
    E_FP_NOM = xp.array(rmad_vars['E_FP_NOM'])
    control_mask = xp.array(rmad_vars['control_mask'])
    wavelength = rmad_vars['wavelength']
    r_cond = rmad_vars['r_cond']

    E_ab_l2norm = E_ab[control_mask].dot(E_ab[control_mask].conjugate()).real

    # Compute E_DM using the forward DM model
    E_FP_with_delA, E_EP, DM_PHASOR = M.forward(
        current_acts + del_acts, 
        wavelength, 
        use_vortex=True, 
        return_ints=True,
    )
    deltaE = E_FP_with_delA - E_FP_NOM

    # compute the cost function
    E_predicted = E_ab + deltaE # take the measured E-field and add the model-based deltaE from new actuator command
    E_predicted_vec = E_predicted[control_mask] # make sure to do array indexing
    J_delE = E_predicted_vec.dot(E_predicted_vec.conjugate()).real
    J_c = r_cond * del_acts_waves.dot(del_acts_waves)
    J = (J_delE + J_c) / E_ab_l2norm
    if verbose: 
        print(f'\tCost-function J_delE: {J_delE:.2e}')
        print(f'\tCost-function J_c: {J_c:.2e}')
        print(f'\tCost-function normalization factor: {E_ab_l2norm:.2e}')
        print(f'\tTotal cost-function value: {J:.2e}\n')

    # Compute the gradient with the adjoint model
    # delE_masked = control_mask * delE # still a 2D array
    # delE_masked = xcipy.ndimage.rotate(delE_masked, -M.det_rotation, reshape=False, order=5)
    # dJ_dE_delA = 2 * delE_masked / E_ab_l2norm
    E_predicted_masked = control_mask * E_predicted # still a 2D array
    E_predicted_masked = xcipy.ndimage.rotate(E_predicted_masked, -M.camsci_rotation, reshape=False, order=5)
    dJ_ddeltaE = 2 * E_predicted_masked / E_ab_l2norm

    camsci_pxscl_lamD = M.camsci_pxscl_lamDc * M.wavelength_c/wavelength
    dJ_dE_FFFP = props.mft_reverse(dJ_ddeltaE, camsci_pxscl_lamD, M.npix * M.lyot_ratio, 2*M.Ndef, convention='+')

    if M.exit_pupil_prop_dist is not None:
        # dJ_dE_FFFP = utils.pad_or_crop(dJ_dE_FFFP, 2*M.Ndef)
        dJ_dE_LS = props.ang_spec(dJ_dE_FFFP, wavelength, -M.exit_pupil_prop_dist, M.exit_pupil_pxscl)
        dJ_dE_LS = utils.pad_or_crop(dJ_dE_LS, M.Ndef)
    else:
        dJ_dE_LS = copy.copy(dJ_dE_FFFP)
        dJ_dE_LS = utils.pad_or_crop(dJ_dE_LS, M.Ndef)

    dJ_dE_LP = dJ_dE_LS * M.LYOTSTOP.astype(xp.complex128)
    # if M.flip_lyot: dJ_dE_LP = xp.fliplr(dJ_dE_LP)
    # if M.reverse_lyot: dJ_dE_LP = xp.rot90(xp.rot90(dJ_dE_LP))

    # Now we have to split and back-propagate the gradient along the two branches used to model the vortex.
    # So one branch for the FFT vortex procedure and one for the MFT vortex procedure. 
    dJ_dE_LP_fft = utils.pad_or_crop(copy.copy(dJ_dE_LP), M.N_vortex_lres)
    dJ_dE_FPM_fft = props.fft(dJ_dE_LP_fft)
    dJ_dE_FP_fft = M.vortex_lres.conj() * (1 - M.lres_window) * dJ_dE_FPM_fft
    dJ_dE_PUP_fft = props.ifft(dJ_dE_FP_fft)
    dJ_dE_PUP_fft = utils.pad_or_crop(dJ_dE_PUP_fft, M.Ndef)

    dJ_dE_FPM_mft = props.mft_forward(dJ_dE_LP,  M.npix, M.N_vortex_hres, M.hres_sampling, convention='-', pp_centering='odd', fp_centering='odd', )
    dJ_dE_FP_mft = M.vortex_hres.conj() * M.hres_window * M.hres_dot_mask * dJ_dE_FPM_mft
    dJ_dE_PUP_mft = props.mft_reverse(dJ_dE_FP_mft, M.hres_sampling, M.npix, M.Ndef, convention='+', pp_centering='odd', fp_centering='odd', )

    dJ_dE_PUP = dJ_dE_PUP_fft + dJ_dE_PUP_mft

    dJ_dS_DM = 4*xp.pi / wavelength * xp.imag(dJ_dE_PUP * E_EP.conj() * utils.pad_or_crop(DM_PHASOR.conj(), M.Ndef))
    # if M.flip_dm: dJ_dS_DM = xp.rot90(xp.rot90(dJ_dS_DM))

    # Now pad back to the array size fo the DM surface to back propagate through the adjoint DM model
    dJ_dS_DM = utils.pad_or_crop(dJ_dS_DM, M.Nsurf)
    x2_bar = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dS_DM.real)))
    x1_bar = M.inf_fun_fft.conjugate() * x2_bar
    dJ_dA = M.Mx_dm_back@x1_bar@M.My_dm_back / ( M.Nsurf * M.Nact * M.Nact ) # why I have to divide by this constant is beyond me

    dJ_dA_vec = dJ_dA[M.dm_mask].real + xp.array( r_cond * 2*del_acts_waves )

    if plot: 
        utils.imshow(
            [xp.abs(dJ_ddeltaE)**2, xp.angle(dJ_ddeltaE),
            xp.abs(dJ_dE_FFFP), xp.angle(dJ_dE_FFFP),
            xp.abs(dJ_dE_LS), xp.angle(dJ_dE_LS),
            xp.abs(dJ_dE_LP), xp.angle(dJ_dE_LP),
            xp.abs(dJ_dE_PUP_fft), xp.angle(dJ_dE_PUP_fft),
            xp.abs(dJ_dE_PUP_mft), xp.angle(dJ_dE_PUP_mft),
            xp.abs(dJ_dE_PUP), xp.angle(dJ_dE_PUP),
            xp.real(dJ_dS_DM), xp.imag(dJ_dS_DM),
            xp.real(dJ_dA), xp.imag(dJ_dA),], 
            titles=[
                'EP Amplitude', 'EP Phase',
            ],
            # npix=8*[int(1.2*self.npix)], 
            cmaps=8*['plasma', 'twilight'],
            norms=[LogNorm(xp.max(xp.abs(dJ_ddeltaE)**2)/1e5)],
            Nrows=9, Ncols=2,
            figsize=(12,48),
            hspace=0.2, wspace=0.2, 
        )

    return ensure_np_array(J), ensure_np_array(dJ_dA_vec)

def val_and_grad_bb(
        del_acts, 
        M, 
        actuators, 
        E_abs, 
        control_mask, 
        waves, 
        r_cond, 
        weights=None, 
        verbose=False, 
        plot=False, 
        fancy_plot=False, 
    ):
    # del_acts, M, actuators, E_ab, control_mask, wavelength, r_cond,
    Nwaves = len(waves)

    del_acts_waves = del_acts/M.wavelength_c

    r_cond_mono = 0
    J_monos = np.zeros(Nwaves)
    dJ_dA_monos = np.zeros((Nwaves, M.Nacts))
    for i in range(Nwaves):
        wavelength = waves[i]
        E_ab = E_abs[i]
        J_mono, dJ_dA_mono = val_and_grad(
            del_acts, 
            M, 
            actuators, 
            E_ab, 
            control_mask, 
            wavelength, 
            r_cond_mono, 
            verbose=verbose, 
            plot=plot, 
            fancy_plot=fancy_plot,
        )
        
        J_monos[i] = J_mono
        dJ_dA_monos[i] = dJ_dA_mono

    J_bb = np.sum(J_monos)/Nwaves + r_cond * del_acts_waves.dot(del_acts_waves)
    dJ_dA_bb = np.sum(dJ_dA_monos, axis=0) + ensure_np_array( r_cond * 2*del_acts_waves )
    
    return J_bb, dJ_dA_bb


def dm_val_and_grad(
        del_acts, 
        OPD,
        M, 
        current_acts=None,
    ):

    del_acts = xp.array(del_acts)
    del_command = xp.zeros((M.Nact, M.Nact))
    del_command[M.dm_mask] = xp.array(del_acts)

    current_acts = xp.array(current_acts) if current_acts is not None else xp.zeros((M.Nact, M.Nact))

    OPD = xp.array(OPD)

    dm_command = current_acts + del_command
    dm_mft = M.Mx_dm@dm_command@M.My_dm
    dm_surf_fft = M.inf_fun_fft * dm_mft
    dm_surf = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(dm_surf_fft,))).real
    dm_surf = utils.pad_or_crop(dm_surf, OPD.shape[0])

    OPD_MASK = utils.pad_or_crop(M.BAP_MASK, OPD.shape[0])
    opd_l2norm = OPD[OPD_MASK].dot(OPD[OPD_MASK])
    total_opd =  OPD + 2*dm_surf
    J = total_opd[OPD_MASK].dot(total_opd[OPD_MASK]) / opd_l2norm
    # print(J)

    masked_total = OPD_MASK * total_opd
    dJ_dOPD = 2 * (masked_total) / opd_l2norm

    dJ_dS_DM = utils.pad_or_crop(dJ_dOPD, M.Nsurf)
    x2_bar = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dS_DM)))
    x1_bar = x2_bar * M.inf_fun_fft.conj()
    dJ_dA1 = M.Mx_dm_back@x1_bar@M.My_dm_back / ( M.Nsurf * M.Nact * M.Nact )

    dJ_dA = dJ_dA1[M.dm_mask].real

    return ensure_np_array(J), ensure_np_array(dJ_dA)


