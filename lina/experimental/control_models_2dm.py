from .math_module import xp, xcipy, ensure_np_array
from aefc_vortex import utils
from aefc_vortex.imshows import imshow1, imshow2, imshow3
from aefc_vortex import dm
from aefc_vortex import props

import numpy as np
import astropy.units as u
from astropy.io import fits

import os
from pathlib import Path
import time
import copy

import poppy

import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec

from scipy.signal import windows
from scipy.optimize import minimize

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
            npix=512,
            Ndef=None,
            N_vortex_lres=2048,
            vortex_win_diam=30, # diameter of the Tukey window in lambda/D to apply for the vortex model 
            vortex_hres_sampling=0.025, # lam/D per pixel; this value is chosen empirically
            vortex_dot_mask_diam_lamDc=0.5,
            dm_beam_diam=9.3e-3,
            z_dm1_dm2=500e-3,
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
        self.z_dm1_dm2 = z_dm1_dm2
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
        self.Ndef = npix + 2 if Ndef is None else Ndef
        self.def_oversample = self.Ndef / self.npix
        self.Nprop = 2*npix
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

        self.dm1_commands = xp.zeros((10, self.Nact, self.Nact))
        self.dm2_commands = xp.zeros((10, self.Nact, self.Nact))
    
    def forward(
            self, 
            actuators, 
            wavelength=None, 
            use_vortex=True, 
            return_ints=False, 
            plot=False,
        ):

        if wavelength is None: wavelength = self.wavelength_c

        dm1_command = xp.zeros((self.Nact,self.Nact))
        dm1_command[self.dm_mask] = xp.array(actuators[:self.Nacts])
        mft_command = self.Mx_dm @ dm1_command @ self.My_dm
        fourier_surf = self.inf_fun_fft * mft_command
        dm_surf = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(fourier_surf,))).real
        DM1_PHASOR = xp.exp(1j * 4*xp.pi/wavelength * dm_surf)
        # DM1_PHASOR = utils.pad_or_crop(DM1_PHASOR, self.Ndef)

        dm2_command = xp.zeros((self.Nact,self.Nact))
        dm2_command[self.dm_mask] = xp.array(actuators[self.Nacts:])
        mft_command = self.Mx_dm @ dm2_command @ self.My_dm
        fourier_surf = self.inf_fun_fft * mft_command
        dm_surf = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(fourier_surf,))).real
        DM2_PHASOR = xp.exp(1j * 4*xp.pi/wavelength * dm_surf)
        # DM2_PHASOR = utils.pad_or_crop(DM2_PHASOR, self.Ndef)
        # if self.flip_dm: DM_PHASOR = xp.rot90(xp.rot90(DM_PHASOR))

        # Initialize the wavefront
        WFE =  self.PREFPM_AMP * xp.exp(1j * 2*xp.pi/wavelength * self.PREFPM_OPD)
        E_EP = self.APERTURE.astype(xp.complex128) * WFE / xp.sqrt(self.Imax_ref)
        E_EP = utils.pad_or_crop(E_EP, self.Nprop)

        E_DM1 = E_EP * utils.pad_or_crop(DM1_PHASOR, self.Nprop)

        E_DM2P = props.ang_spec(E_DM1, wavelength, self.z_dm1_dm2, self.dm_pxscl)

        E_DM2 = E_DM2P * utils.pad_or_crop(DM2_PHASOR, self.Nprop)

        E_PUP = props.ang_spec(E_DM2, wavelength, -self.z_dm1_dm2, self.dm_pxscl)

        if use_vortex:
            E_DM_lres = utils.pad_or_crop(E_PUP, self.N_vortex_lres) # pad to the larger array for the low res propagation
            E_FPM_lres = props.fft(E_DM_lres)
            E_FPM_lres *= self.windowed_vortex_lres
            E_LP_lres = props.ifft(E_FPM_lres)
            E_LP_lres = utils.pad_or_crop(E_LP_lres, self.Ndef) # crop to the desired wavefront dimension

            E_FPM_hres = props.mft_forward(
                E_PUP, 
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
            E_LP = copy.copy(E_PUP)

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
            return E_FP, E_EP, DM1_PHASOR, E_DM2P, DM2_PHASOR
        else:
            return E_FP
        
    def forward_mw(
            self, 
            actuators,
            waves, 
            wavelength=None, 
            use_vortex=True, 
            return_ints=False, 
        ):
        E_FPs = []
        for i in range(len(waves)):
            E_FPs.append(self.forward(actuators, waves[i], use_vortex=True, return_ints=False))
        E_FPs = xp.array(E_FPs)
        return E_FPs

    def calc_wf(self):
        dm_command = xp.sum(self.dm_commands, axis=0)
        E_FP = self.forward(dm_command[self.dm_mask], self.wavelength, self.use_vortex)
        return E_FP

    def snap(self):
        dm_command = xp.sum(self.dm_commands, axis=0)
        E_FP = self.forward(dm_command[self.dm_mask], self.wavelength, self.use_vortex)
        im = xp.abs(E_FP)**2
        return im
    
    def snap_bb(self, waves):
        Nwaves = len(waves)
        im = 0.0
        for i in range(Nwaves):
            self.wavelength = waves[i]
            im += self.snap()/Nwaves
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
    wfs_mask = xp.array(rmad_vars['wfs_mask'])
    wavelength = rmad_vars['wavelength']
    r_cond = rmad_vars['r_cond']

    E_ab_l2norm = E_ab[wfs_mask].dot(E_ab[wfs_mask].conjugate()).real

    # Compute E_DM using the forward DM model
    E_FP_with_delA, E_EP, DM1_PHASOR, E_DM2P, DM2_PHASOR  = M.forward(
        current_acts + del_acts, 
        wavelength, 
        use_vortex=True, 
        return_ints=True,
    )
    deltaE = E_FP_with_delA - E_FP_NOM

    # compute the cost function
    E_predicted = E_ab + deltaE # take the measured E-field and add the model-based deltaE from new actuator command
    E_predicted_vec = E_predicted[wfs_mask] # make sure to do array indexing
    J_delE = E_predicted_vec.dot(E_predicted_vec.conjugate()).real
    J_c = r_cond * del_acts_waves.dot(del_acts_waves)
    J = (J_delE + J_c) / E_ab_l2norm
    if verbose: 
        print(f'\tCost-function J_delE: {J_delE:.2e}')
        print(f'\tCost-function J_c: {J_c:.2e}')
        print(f'\tCost-function normalization factor: {E_ab_l2norm:.2e}')
        print(f'\tTotal cost-function value: {J:.2e}\n')

    # Compute the gradient with the adjoint model
    # delE_masked = wfs_mask * delE # still a 2D array
    # delE_masked = xcipy.ndimage.rotate(delE_masked, -M.det_rotation, reshape=False, order=5)
    # dJ_dE_delA = 2 * delE_masked / E_ab_l2norm
    E_predicted_masked = wfs_mask * E_predicted # still a 2D array
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
    dJ_dE_PUP_fft = utils.pad_or_crop(dJ_dE_PUP_fft, M.Nprop)

    dJ_dE_FPM_mft = props.mft_forward(dJ_dE_LP,  M.npix, M.N_vortex_hres, M.hres_sampling, convention='-', pp_centering='odd', fp_centering='odd', )
    dJ_dE_FP_mft = M.vortex_hres.conj() * M.hres_window * M.hres_dot_mask * dJ_dE_FPM_mft
    dJ_dE_PUP_mft = props.mft_reverse(dJ_dE_FP_mft, M.hres_sampling, M.npix, M.Nprop, convention='+', pp_centering='odd', fp_centering='odd', )

    dJ_dE_PUP = dJ_dE_PUP_fft + dJ_dE_PUP_mft

    dJ_dE_DM2 = props.ang_spec(dJ_dE_PUP, wavelength, M.z_dm1_dm2, M.dm_pxscl)

    dJ_dE_DM2P = dJ_dE_DM2 * DM2_PHASOR.conj()

    dJ_dE_DM1 = props.ang_spec(dJ_dE_DM2P, wavelength, -M.d_dm1_dm2, M.dm_pxscl)

    dJ_dS_DM2 = 4*xp.pi/wavelength * xp.imag(dJ_dE_DM2 * E_DM2P.conj() * DM2_PHASOR.conj())
    dJ_dS_DM1 = 4*xp.pi/wavelength * xp.imag(dJ_dE_DM1 * E_EP.conj() * DM1_PHASOR.conj())
    if M.flip_dm: 
        dJ_dS_DM2 = xp.rot90(xp.rot90(dJ_dS_DM2))
        dJ_dS_DM1 = xp.rot90(xp.rot90(dJ_dS_DM1))

    # Now pad back to the array size fo the DM surface to back propagate through the adjoint DM model
    dJ_dS_DM2 = utils.pad_or_crop(dJ_dS_DM2, M.Nsurf)
    x2_bar = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dS_DM2.real)))
    x1_bar = M.inf_fun_fft.conjugate() * x2_bar
    dJ_dA2 = M.Mx_dm_back@x1_bar@M.My_dm_back / ( M.Nsurf * M.Nact * M.Nact ) 

    dJ_dS_DM1 = utils.pad_or_crop(dJ_dS_DM1, M.Nsurf)
    x2_bar = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dS_DM1.real)))
    x1_bar = M.inf_fun_fft.conjugate() * x2_bar
    dJ_dA1 = M.Mx_dm_back@x1_bar@M.My_dm_back / ( M.Nsurf * M.Nact * M.Nact ) # why I have to divide by this constant is beyond me

    dJ_dA_vec = dJ_dA[M.dm_mask].real + xp.array( r_cond * 2*del_acts_waves )
    dJ_dA = xp.concatenate([dJ_dA1[M.dm_mask].real, dJ_dA2[M.dm_mask].real]) + xp.array( r_cond * 2*del_acts_waves )

    if plot: 
        utils.imshow(
            [xp.abs(dJ_ddeltaE)**2, xp.angle(dJ_ddeltaE),
            xp.abs(dJ_dE_FFFP), xp.angle(dJ_dE_FFFP),
            xp.abs(dJ_dE_LS), xp.angle(dJ_dE_LS),
            xp.abs(dJ_dE_LP), xp.angle(dJ_dE_LP),
            xp.abs(dJ_dE_PUP_fft), xp.angle(dJ_dE_PUP_fft),
            xp.abs(dJ_dE_PUP_mft), xp.angle(dJ_dE_PUP_mft),
            xp.abs(dJ_dE_PUP), xp.angle(dJ_dE_PUP),
            xp.real(dJ_dS_DM2), xp.imag(dJ_dS_DM2),
            xp.real(dJ_dS_DM1), xp.imag(dJ_dS_DM1),
            xp.real(dJ_dA2), xp.imag(dJ_dA2),
            xp.real(dJ_dA1), xp.imag(dJ_dA1),], 
            titles=[
                'Intensity', 'Phase',
                'Amplitude', 'Phase',
                'Amplitude', 'Phase',
                'Amplitude', 'Phase',
                'Amplitude', 'Phase',
                'Amplitude', 'Phase',
                'Amplitude', 'Phase',
                'Real', 'Imaginary',
                'Real', 'Imaginary',
                'Real', 'Imaginary',
                'Real', 'Imaginary',
            ],
            # npix=8*[int(1.2*self.npix)], 
            cmaps=12*['plasma', 'twilight'],
            norms=[LogNorm(xp.max(xp.abs(dJ_ddeltaE)**2)/1e5)],
            Nrows=11, Ncols=2,
            figsize=(12,60),
            hspace=0.2, wspace=0.2, 
        )

    return ensure_np_array(J), ensure_np_array(dJ_dA_vec)

def val_and_grad(
        del_acts, 
        M, 
        rmad_vars,
        verbose=False, 
        plot=False, 
        fancy_plot=False,
    ):
    # Convert array arguments into correct types
    del_acts = xp.array(del_acts)
    del_acts_waves = del_acts/M.wavelength_c

    # Not normalizing by the strongest actuator, but that could be changed to use 
    
    current_acts = rmad_vars['current_acts']
    E_ab = rmad_vars['E_ab']
    E_FP_NOM = rmad_vars['E_FP_NOM']
    wavelength = rmad_vars['wavelength']
    control_mask = rmad_vars['control_mask']
    r_cond = rmad_vars['r_cond']
    
    E_ab_l2norm = E_ab[control_mask].dot(E_ab[control_mask].conjugate()).real

    # Compute E_dm using the forward DM model
    E_FP_with_delA, E_EP, E_DM2P, DM1_PHASOR, DM2_PHASOR = M.forward(
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
        print(f'\tCost-function J_delE: {J_delE:.3f}')
        print(f'\tCost-function J_c: {J_c:.3f}')
        print(f'\tCost-function normalization factor: {E_ab_l2norm:.3f}')
        print(f'\tTotal cost-function value: {J:.3f}\n')

    # Compute the gradient with the adjoint model
    E_predicted_masked = control_mask * E_predicted # still a 2D array
    dJ_ddeltaE = 2 * E_predicted_masked / E_ab_l2norm

    psf_pixelscale_lamD = M.psf_pixelscale_lamDc * M.wavelength_c/wavelength
    dJ_dE_LS = props.mft_reverse(dJ_ddeltaE, psf_pixelscale_lamD, M.npix * M.lyot_ratio, M.N, convention='+')
    if plot: imshow2(xp.abs(dJ_dE_LS), xp.angle(dJ_dE_LS), 'RMAD Lyot Stop', npix=1.5*M.npix)

    dJ_dE_LP = dJ_dE_LS * utils.pad_or_crop(M.LYOT, M.N)
    if M.flip_lyot: 
        dJ_dE_LP = xp.rot90(xp.rot90(dJ_dE_LP))
    if plot: imshow2(xp.abs(dJ_dE_LP), xp.angle(dJ_dE_LP), 'RMAD Lyot Pupil', npix=1.5*M.npix)

    # Now we have to split and back-propagate the gradient along the two branches used to model 
    # the vortex. So one branch for the FFT vortex procedure and one for the MFT vortex procedure. 
    dJ_dE_LP_fft = utils.pad_or_crop(copy.copy(dJ_dE_LP), M.N_vortex_lres)
    dJ_dE_FPM_fft = props.fft(dJ_dE_LP_fft)
    dJ_dE_FP_fft = M.vortex_lres.conjugate() * (1 - M.lres_window) * dJ_dE_FPM_fft
    dJ_dE_PUP_fft = props.ifft(dJ_dE_FP_fft)
    dJ_dE_PUP_fft = utils.pad_or_crop(dJ_dE_PUP_fft, M.N)
    if plot: imshow2(xp.abs(dJ_dE_PUP_fft), xp.angle(dJ_dE_PUP_fft), 'RMAD FFT Pupil', npix=1.5*M.npix)

    dJ_dE_LP_mft = utils.pad_or_crop(copy.copy(dJ_dE_LP), M.N)
    dJ_dE_FPM_mft = props.mft_forward(dJ_dE_LP_mft,  M.npix, M.N_vortex_hres, M.hres_sampling, convention='-')
    dJ_dE_FP_mft = M.vortex_hres.conjugate() * M.hres_window * M.hres_dot_mask * dJ_dE_FPM_mft
    dJ_dE_PUP_mft = props.mft_reverse(dJ_dE_FP_mft, M.hres_sampling, M.npix, M.N, convention='+')
    if plot: imshow2(xp.abs(dJ_dE_PUP_mft), xp.angle(dJ_dE_PUP_mft), 'RMAD MFT Pupil', npix=1.5*M.npix)

    dJ_dE_PUP = dJ_dE_PUP_fft + dJ_dE_PUP_mft
    if plot: imshow2(xp.abs(dJ_dE_PUP), xp.angle(dJ_dE_PUP), 'RMAD Total Pupil', npix=1.5*M.npix)

    dJ_dE_DM2 = props.ang_spec(dJ_dE_PUP, wavelength*u.m, M.d_dm1_dm2, M.dm_pxscl)
    if plot: imshow2(xp.abs(dJ_dE_DM2), xp.angle(dJ_dE_DM2), 'RMAD DM2 WF', npix=1.5*M.npix)

    dJ_dE_DM2P = dJ_dE_DM2 * DM2_PHASOR.conj()
    if plot: imshow2(xp.abs(dJ_dE_DM2P), xp.angle(dJ_dE_DM2P), 'RMAD DM2 Plane WF', npix=1.5*M.npix)

    dJ_dE_DM1 = props.ang_spec(dJ_dE_DM2P, wavelength*u.m, -M.d_dm1_dm2, M.dm_pxscl)
    if plot: imshow2(xp.abs(dJ_dE_DM1), xp.angle(dJ_dE_DM1), 'RMAD DM1 WF', npix=1.5*M.npix)

    dJ_dS_DM2 = 4*xp.pi/wavelength * xp.imag(dJ_dE_DM2 * E_DM2P.conj() * DM2_PHASOR.conj())
    dJ_dS_DM1 = 4*xp.pi/wavelength * xp.imag(dJ_dE_DM1 * E_EP.conj() * DM1_PHASOR.conj())
    if M.flip_dm: 
        dJ_dS_DM1 = xp.rot90(xp.rot90(dJ_dS_DM1))
        dJ_dS_DM2 = xp.rot90(xp.rot90(dJ_dS_DM2))
    if plot: imshow2(xp.real(dJ_dS_DM2), xp.imag(dJ_dS_DM2), 'RMAD DM2 Surface', npix=1.5*M.npix)
    if plot: imshow2(xp.real(dJ_dS_DM1), xp.imag(dJ_dS_DM1), 'RMAD DM1 Surface', npix=1.5*M.npix)

    # Now pad back to the array size fo the DM surface to back propagate through the adjoint DM model
    dJ_dS_DM2 = utils.pad_or_crop(dJ_dS_DM2, M.Nsurf)
    x2_bar = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dS_DM2)))
    x1_bar = x2_bar * M.inf_fun_fft.conj()
    dJ_dA2 = M.Mx_back@x1_bar@M.My_back / ( M.Nsurf * M.Nact * M.Nact ) # why I have to divide by this constant is beyond me
    if plot: imshow2(dJ_dA2.real, dJ_dA2.imag, 'RMAD DM2 Actuators')

    dJ_dS_DM1 = utils.pad_or_crop(dJ_dS_DM1, M.Nsurf)
    x2_bar = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dS_DM1)))
    x1_bar = x2_bar * M.inf_fun_fft.conj()
    dJ_dA1 = M.Mx_back@x1_bar@M.My_back / ( M.Nsurf * M.Nact * M.Nact ) # why I have to divide by this constant is beyond me
    if plot: imshow2(dJ_dA1.real, dJ_dA1.imag, 'RMAD DM1 Actuators')

    dJ_dA = xp.concatenate([dJ_dA1[M.dm_mask].real, dJ_dA2[M.dm_mask].real]) + xp.array( r_cond * 2*del_acts_waves )

    if fancy_plot: 
        fancy_plot_adjoint(dJ_ddeltaE, dJ_dE_LP, dJ_dE_PUP, dJ_dS_DM1, dJ_dS_DM2, dJ_dA1, dJ_dA2, control_mask)

    return ensure_np_array(J), ensure_np_array(dJ_dA)



# class MODEL():
#     def __init__(
#             self,
#             npix=1000,
#         ):

#         # initialize physical parameters
#         self.wavelength_c = 650e-9
#         self.dm_beam_diam = 47*u.mm
#         self.d_dm1_dm2 = 700*u.mm
#         # self.d_dm1_dm2 = 0*u.mm
#         self.lyot_pupil_diam = 4/5*self.dm_beam_diam
#         self.lyot_stop_diam = 0.9 * self.lyot_pupil_diam
#         self.lyot_ratio = 0.9
#         self.control_rad = 96/2 * 47/48 * self.lyot_ratio
#         self.psf_pixelscale_lamDc = 0.347
#         self.psf_pixelscale_lamD = self.psf_pixelscale_lamDc
#         self.npsf = 256

#         self.Imax_ref = 1

#         # initialize sampling parameters and load masks
#         self.npix = npix
#         self.oversample = 4.096
#         self.N = int(self.npix*self.oversample)

#         pwf = poppy.FresnelWavefront(beam_radius=self.dm_beam_diam/2, npix=self.npix, oversample=1) # pupil wavefront
#         self.APERTURE = poppy.CircularAperture(radius=self.dm_beam_diam/2).get_transmission(pwf)
#         self.APMASK = self.APERTURE>0
#         self.LYOT = poppy.CircularAperture(radius=self.lyot_ratio*self.dm_beam_diam/2).get_transmission(pwf)
#         self.AMP = xp.ones((self.npix,self.npix))
#         self.OPD = xp.zeros((self.npix,self.npix))

#         self.Nact = 96
#         self.act_spacing = 500e-6*u.m
#         self.dm_pxscl = self.dm_beam_diam/(self.npix * u.pix)
#         self.inf_sampling = self.act_spacing.to_value(u.m)/self.dm_pxscl.to_value(u.m/u.pix)
#         self.inf_fun = dm.make_gaussian_inf_fun(act_spacing=self.act_spacing, sampling=self.inf_sampling, coupling=0.15, Nact=self.Nact+2)
#         self.Nsurf = self.inf_fun.shape[0]

#         y,x = (xp.indices((self.Nact, self.Nact)) - self.Nact//2 + 1/2)
#         r = xp.sqrt(x**2 + y**2)
#         self.dm_mask = r<(self.Nact/2 + 1/2)
#         self.Nacts = int(2*self.dm_mask.sum())

#         self.inf_fun_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(self.inf_fun,)))
#         # DM command coordinates
#         xc = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)
#         yc = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)

#         # Influence function frequncy sampling
#         fx = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))
#         fy = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))

#         # forward DM model MFT matrices
#         self.Mx = xp.exp(-1j*2*np.pi*xp.outer(fx,xc))
#         self.My = xp.exp(-1j*2*np.pi*xp.outer(yc,fy))

#         self.Mx_back = xp.exp(1j*2*np.pi*xp.outer(xc,fx))
#         self.My_back = xp.exp(1j*2*np.pi*xp.outer(fy,yc))

#         # Vortex model parameters
#         self.oversample_vortex = 4.096
#         self.N_vortex_lres = int(self.npix*self.oversample_vortex)
#         self.lres_sampling = 1/self.oversample_vortex # low resolution sampling in lam/D per pixel
#         self.lres_win_size = int(30/self.lres_sampling)
#         w1d = xp.array(windows.tukey(self.lres_win_size, 1, False))
#         self.lres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_lres)
#         self.vortex_lres = props.make_vortex_phase_mask(self.N_vortex_lres)

#         self.hres_sampling = 0.025 # lam/D per pixel; this value is chosen empirically
#         self.N_vortex_hres = int(np.round(30/self.hres_sampling))
#         self.hres_win_size = int(30/self.hres_sampling)
#         w1d = xp.array(windows.tukey(self.hres_win_size, 1, False))
#         self.hres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_hres)
#         self.vortex_hres = props.make_vortex_phase_mask(self.N_vortex_hres)

#         y,x = (xp.indices((self.N_vortex_hres, self.N_vortex_hres)) - self.N_vortex_hres//2)*self.hres_sampling
#         r = xp.sqrt(x**2 + y**2)
#         self.hres_dot_mask = r>=0.15

#         self.det_rotation = 0
#         self.flip_dm = False
#         self.flip_lyot = False

#         self.use_vortex = True

#         self.dm1_command = xp.zeros((self.Nact, self.Nact))
#         self.dm2_command = xp.zeros((self.Nact, self.Nact))

#     def forward(self, actuators, wavelength, use_vortex=True, return_ints=False, plot=False, fancy_plot=False):
#         dm1_command = xp.zeros((self.Nact,self.Nact))
#         dm1_command[self.dm_mask] = xp.array(actuators[:self.Nacts//2])
#         dm1_mft = self.Mx@dm1_command@self.My
#         dm1_surf_fft = self.inf_fun_fft * dm1_mft
#         dm1_surf = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(dm1_surf_fft,))).real
#         dm1_surf = utils.pad_or_crop(dm1_surf, self.N)
#         DM1_PHASOR = xp.exp(1j * 4*xp.pi/wavelength * dm1_surf)

#         dm2_command = xp.zeros((self.Nact,self.Nact))
#         dm2_command[self.dm_mask] = xp.array(actuators[self.Nacts//2:])
#         dm2_mft = self.Mx@dm2_command@self.My
#         dm2_surf_fft = self.inf_fun_fft * dm2_mft
#         dm2_surf = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(dm2_surf_fft,))).real
#         dm2_surf = utils.pad_or_crop(dm2_surf, self.N)
#         DM2_PHASOR = xp.exp(1j * 4*xp.pi/wavelength * dm2_surf)

#         if self.flip_dm: 
#             DM1_PHASOR = xp.rot90(xp.rot90(DM1_PHASOR))
#             DM2_PHASOR = xp.rot90(xp.rot90(DM2_PHASOR))

#         # Initialize the wavefront
#         WFE = utils.pad_or_crop(self.AMP, self.N) * xp.exp(1j * 2*xp.pi/wavelength * utils.pad_or_crop(self.OPD, self.N))
#         E_EP = utils.pad_or_crop(self.APERTURE.astype(xp.complex128), self.N) * WFE / xp.sqrt(self.Imax_ref)
#         if plot: imshow2(xp.abs(E_EP), xp.angle(E_EP), 'Entrance Pupil WF', npix=1.5*self.npix)

#         E_DM1 = E_EP * utils.pad_or_crop(DM1_PHASOR, self.N)
#         if plot: imshow2(xp.abs(E_DM1), xp.angle(E_DM1), 'After DM1 WF', npix=1.5*self.npix)

#         E_DM2P = props.ang_spec(E_DM1, wavelength*u.m, self.d_dm1_dm2, self.dm_pxscl)
#         if plot: imshow2(xp.abs(E_DM2P), xp.angle(E_DM2P), 'At DM2 WF', npix=1.5*self.npix)

#         E_DM2 = E_DM2P * utils.pad_or_crop(DM2_PHASOR, self.N)
#         if plot: imshow2(xp.abs(E_DM2), xp.angle(E_DM2), 'After DM2 WF', npix=1.5*self.npix)

#         E_PUP = props.ang_spec(E_DM2, wavelength*u.m, -self.d_dm1_dm2, self.dm_pxscl)
#         if plot: imshow2(xp.abs(E_PUP), xp.angle(E_PUP), 'Back to Pupil WF', npix=1.5*self.npix)

#         if use_vortex:
#             lres_wf = utils.pad_or_crop(E_PUP, self.N_vortex_lres) # pad to the larger array for the low res propagation
#             fp_wf_lres = props.fft(lres_wf)
#             fp_wf_lres *= self.vortex_lres * (1 - self.lres_window) # apply low res FPM and inverse Tukey window
#             pupil_wf_lres = props.ifft(fp_wf_lres)
#             pupil_wf_lres = utils.pad_or_crop(pupil_wf_lres, self.N)
#             if plot: imshow2(xp.abs(pupil_wf_lres), xp.angle(pupil_wf_lres), 'Vortex FFT WF', npix=1.5*self.npix)

#             fp_wf_hres = props.mft_forward(E_PUP, self.npix, self.N_vortex_hres, self.hres_sampling, convention='-')
#             fp_wf_hres *= self.vortex_hres * self.hres_window * self.hres_dot_mask # apply high res FPM, window, and dot mask
#             pupil_wf_hres = props.mft_reverse(fp_wf_hres, self.hres_sampling, self.npix, self.N, convention='+')
#             if plot: imshow2(xp.abs(pupil_wf_hres), xp.angle(pupil_wf_hres), 'Vortex MFT WF', npix=1.5*self.npix)

#             E_LP = (pupil_wf_lres + pupil_wf_hres)
#             if plot: imshow2(xp.abs(E_LP), xp.angle(E_LP), 'Post Vortex Relay Pupil WF', npix=1.5*self.npix)
#         else:
#             E_LP = E_PUP
        
#         if self.flip_lyot: 
#             E_LP = xp.rot90(xp.rot90(E_LP))

#         E_LS = E_LP * utils.pad_or_crop(self.LYOT, self.N)
#         if plot: imshow2(xp.abs(E_LS), xp.angle(E_LS), 'After Lyot Stop WF', npix=1.5*self.npix)
        
#         psf_pixelscale_lamD = self.psf_pixelscale_lamDc * self.wavelength_c/wavelength
#         E_FP = props.mft_forward(E_LS, self.npix * self.lyot_ratio, self.npsf, psf_pixelscale_lamD)
#         if plot: imshow2(xp.abs(E_FP)**2, xp.angle(E_FP), lognorm1=True)

#         if fancy_plot: 
#             fancy_plot_forward(dm1_command, dm2_command, DM1_PHASOR, DM2_PHASOR, E_PUP, E_LP, E_FP, npix=self.npix, wavelength=wavelength)

#         if return_ints:
#             return E_FP, E_EP, E_DM2P, DM1_PHASOR, DM2_PHASOR
#         else:
#             return E_FP
        
#     def getattr(self, attr):
#         return getattr(self, attr)
    
#     def setattr(self, attr, val):
#         setattr(self, attr, val)
        
#     def zero_dms(self):
#         self.dm1_command = xp.zeros((self.Nact,self.Nact))
#         self.dm2_command = xp.zeros((self.Nact,self.Nact))

#     def add_dm1(self, del_dm):
#         self.dm1_command += del_dm
    
#     def set_dm1(self, dm_command):
#         self.dm1_command = dm_command

#     def get_dm1(self,):
#         return copy.copy(self.dm1_command)

#     def add_dm2(self, del_dm):
#         self.dm2_command += del_dm
    
#     def set_dm2(self, dm_command):
#         self.dm2_command = dm_command

#     def get_dm2(self,):
#         return copy.copy(self.dm2_command)
        
#     def calc_wf(self, wavelength=650e-9):
#         actuators = xp.concatenate([self.dm1_command[self.dm_mask], self.dm2_command[self.dm_mask]])
#         fpwf = self.forward(actuators, wavelength, use_vortex=self.use_vortex, )
#         return fpwf
        
#     def snap(self):
#         actuators = xp.concatenate([self.dm1_command[self.dm_mask], self.dm2_command[self.dm_mask]])
#         im = xp.abs(self.forward(actuators, self.wavelength, use_vortex=self.use_vortex,))**2
#         return im
