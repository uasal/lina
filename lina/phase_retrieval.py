import numpy as truenp
import matplotlib.pyplot as plt

from prysm import (
    mathops, 
    conf,
)
from prysm.mathops import (
    np,
    fft,
    interpolate,
    ndimage,
)
from prysm.coordinates import (
    make_xy_grid, 
    cart_to_polar,
)
from prysm.propagation import Wavefront as WF
from prysm.propagation import (
    focus_fixed_sampling,
    focus_fixed_sampling_backprop
)                              

from prysm.polynomials import (
    noll_to_nm,
    zernike_nm_seq,
    hopkins,
    sum_of_2d_modes
)

from scipy.optimize import minimize


def MSE(I, D):
    return np.sum((I - D) ** 2)


def grad_MSE(I, D):
    return 2 * (I - D)


def GIE(I, D):
    t1 = np.sum(I * D) ** 2
    t2 = np.sum(D ** 2) 
    t3 = np.sum(I ** 2)
    return 1 - t1 / (t2 * t3)


def grad_GIE(I, D):
    t1 = np.sum(I * D)
    t2 = np.sum(D ** 2)
    t3 = np.sum(I ** 2)
    return 2 * t1 / (t2 * t3 ** 2) * (I * t1 - D * t3)

class ADPR:
    def __init__(self, wvls, pupil, dx_pupil, psf, dx_psf, efl, modes, 
                 defocus_coeff, initial_opd=None, error_fx='GIE'):
        
        """A class for performing phase retrieval using algorithmic differentiation on a single PSF. 
        Not terribly useful on its own, but several of these can be fed to as a list to the FDPR or FFPR
        classes to perform focus-diverse or full-field phase retrieval, respectively.

        Args:
            wvls (list): wavelength(s), microns.
            pupil (ndarray): pupil function.
            dx_pupil (float): spacing between samples in the pupil plane, millimeters.
            psf (ndarray): ground-truth PSF image to perform phase retrieval on.
            dx_psf (float): spacing between samples in the focal plane, microns.
            efl (float): effective focal length of the optical system, millimeters.
            modes (list): the modal basis whose coefficients will be optimized for during phase retrieval.
            defocus_coeff (float): the amount of defocus present in the PSF, waves.
            initial_opd (ndarray, optional): initial guess of the pupil OPD, nanometers. Defaults to None.
            error_fx (str, optional): which error function to use. 
            'MSE' is for mean squared error.
            'GIE' is for gain-invariant error; performs better with noisy data.
            Defaults to 'GIE'.
        """

        # ADPR parameters
        self.wvls = wvls
        self.pup = pupil
        self.dx_pup = dx_pupil
        self.D_pup = pupil.shape[0] * dx_pupil
        self.efl = efl
        self.modes = np.array(modes)
        self.psf = psf
        self.dx_psf = dx_psf
        self.error_fx = error_fx.upper()
        self.costs = []

        if initial_opd is None:
            self.init_opd = np.zeros(pupil.shape, dtype=float)
        else:
            self.init_opd = initial_opd

        if error_fx == 'MSE':
            self.err = MSE 
            self.grad_err = grad_MSE
        elif error_fx == 'GIE':
            self.err = GIE
            self.grad_err = grad_GIE
        else:
            self.err = GIE
            self.grad_err = grad_GIE
            print("INVALID ERROR FUNCTION PROVIDED, DEFAULTING TO GAIN INVARIANT ERROR")

        # defocus 
        self.defocus_coeff = defocus_coeff
        self.defocus_coeff *= (wvls[0] + wvls[-1]) / 2 * 1e3 # waves -> um -> nm

        x, y = make_xy_grid(pupil.shape[0], diameter=self.D_pup)
        r, t = cart_to_polar(x, y)
        r_norm = r / (self.D_pup / 2)
        self.W020 = hopkins(0, 2, 0, r_norm, t, 0)
        self.defocus = self.W020 * self.defocus_coeff * pupil

    def fwd(self, x):
        
        # calculate OPD at the pupil using the initial OPD guess, the 
        # modal basis and the modal coefficients
        self.opd = self.init_opd + sum_of_2d_modes(self.modes, np.array(x))

        # initialize: 
        # per-wavelength W: the pupil phase
        # per-wavelength g: the pupil-plane complex wavefront
        # per-wavelength G: the focal-plane complex wavefront
        # I: focal-plane intensity 
        self.Ws = []
        self.gs = []
        self.Gs = []
        self.I = 0

        # loop through wavelengths
        for wvl in self.wvls:
            
            # convert OPD at the pupil to pupil phase, the defocus term is included at this step
            W = (2 * np.pi / wvl) * (self.opd - self.defocus) / 1e3
            self.Ws.append(W)

            # create the pupil-plane complex wavefront using pupil phase and a pupil function
            g = self.pup * np.exp(1j * W)
            self.gs.append(g)

            # propagate the pupil-plane complex wavefront to the focal plane
            # this is just a matrix DFT
            G = focus_fixed_sampling(wavefunction=g, input_dx=self.dx_pup, prop_dist=self.efl, wavelength=wvl,
                                     output_dx=self.dx_psf, output_samples=self.psf.shape[0], shift=(0, 0), method='mdft')
            self.Gs.append(G)

            # convert the focal-plane complex wavefront to intensity, if multiple wavelengths are given 
            # then we sum the monochromatic intensities across wavelengths to get the broadband intensity
            self.I += np.abs(G) ** 2 / len(self.wvls)

        # calculate the error between the estimated PSF and the ground-truth PSF
        self.E = self.err(self.I, self.psf)

        self.costs.append(self.E)

        return self.E
    
    def rev(self):
        
        # initialize:
        # per-wavelength Ibar: the focal-plane intensity gradient with respect to error
        # per-wavelength Gbar: the focal-plane complex wavefront gradient with respect to the focal-plane intensity gradient
        # per-wavelength gbar: the pupil-plane complex wavefront gradient with respect to the focal-plane complex wavefront gradient
        # Wbar: the pupil phase gradient with respect to the pupil-plane complex wavefront gradient
        self.Ibars = []
        self.Gbars = []
        self.gbars = []
        self.Wbar = 0

        # loop through wavelengths, per-wavelength G, and per-wavelength g
        for wvl, G, g in zip(self.wvls, self.Gs, self.gs):
            
            # calculate the focal-plane intensity gradient with respect to error
            Ibar = self.grad_err(self.I, self.psf) / len(self.wvls)
            self.Ibars.append(Ibar)

            # calculate the focal-plane complex wavefront gradient with respect to the focal-plane intensity gradient
            Gbar = 2 * Ibar * G 
            self.Gbars.append(Gbar)

            # calculate the pupil-plane complex wavefront gradient with respect to the focal-plane complex wavefront gradient
            # this is just an inverse matrix-DFT
            gbar = focus_fixed_sampling_backprop(wavefunction=Gbar, input_dx=self.dx_pup, prop_dist=self.efl, wavelength=wvl,
                                                 output_dx=self.dx_psf, output_samples=self.pup.shape[0], shift=(0, 0), method='mdft')
            self.gbars.append(gbar)

            # calculate the pupil phase gradient with respect to the pupil-plane complex wavefront gradient, if
            # multiple wavelengths are given then we sum the gradients across wavelengths to get the total gradient
            self.Wbar += (2 * np.pi / wvl) * np.imag(gbar * np.conj(g)) / 1e3

        # calculate the modal coefficient gradient with respect to the pupil phase gradient
        self.abar = np.tensordot(self.modes, self.Wbar)

        return self.abar
    
    def fg(self, x):

        f = self.fwd(x)

        g = self.rev()

        return f.get(), g.get()

class FDPR:

    def __init__(self, optlist):

        self.optlist = optlist
        self.f = 0
        self.g = 0
        self.costs = []

    def refresh(self):
        self.f = 0
        self.g = 0
    
    def fg(self, x):

        # reset the f, g values
        self.refresh()

        # just sum them
        for opt in self.optlist:
            f, g = opt.fg(x)
            self.f += f
            self.g += g
        
        self.costs.append(self.f)

        return self.f, self.g
    

def ensure_np(arg):
    if isinstance(arg, truenp.ndarray):
        return arg
    if hasattr(arg, 'get'):
        return arg.get()


class FFPR:
    def __init__(self, optlist, psf_positions, field_modes, field_coeff_interps):
        
        # list of individual PSF optimizers
        self.optlist = optlist 

        # psf positions in the field
        self.psf_positions = psf_positions

        # interpolators for Z4 thru Z11 which return coeffs given a field postion
        # units for field position should be consistent with `psf_positions`
        self.field_interps = field_coeff_interps

        # for calculating field-dependent coeff deviations from nominal
        self.Z4_a = 0
        self.Z4_b = 0
        self.Z4_c = 0

        self.Z5_a = 0
        self.Z5_b = 0
        self.Z5_c = 0

        self.Z6_a = 0
        self.Z6_b = 0
        self.Z6_c = 0

        self.Z7_a = 0
        self.Z7_b = 0
        self.Z7_c = 0
        
        self.Z8_a = 0
        self.Z8_b = 0
        self.Z8_c = 0

        self.Z11_a = 0
        self.Z11_b = 0
        self.Z11_c = 0
        
        # for field-dependent optimization
        self.modes_field = field_modes
        self.coeffs_field_nom = [[interp(np.array(position)) for interp in self.field_interps] for position in self.psf_positions]

        # for joint optimization
        self.modes_common = optlist[0].modes
        self.coeffs_common = np.zeros(len(self.modes_common))

        self.costs = []


    def _fwd_calc_coeffs_field(self, position, coeffs_nom):

        coeffs_field = np.zeros(len(self.modes_field))

        # Z4 deviation from nominal varies linearly across the field
        coeffs_field[0] = self.Z4_a * position[0] + self.Z4_b * position[1] + self.Z4_c + coeffs_nom[0]

        # Z5 deviation from nominal varies linearly across the field
        coeffs_field[1] = self.Z5_a * position[0] + self.Z5_b * position[1] + self.Z5_c + coeffs_nom[1]

        # Z6 deviation from nominal varies linearly across the field
        coeffs_field[2] = self.Z6_a * position[0] + self.Z6_b * position[1] + self.Z6_c + coeffs_nom[2]

        # Z7 deviation from nominal varies linearly across the field
        coeffs_field[3] = self.Z7_a * position[0] + self.Z7_b * position[1] + self.Z7_c + coeffs_nom[3]

        # Z8 deviation from nominal varies linearly across the field
        coeffs_field[4] = self.Z8_a * position[0] + self.Z8_b * position[1] + self.Z8_c + coeffs_nom[4]

        # Z9 does not deviate from nominal
        coeffs_field[5] = coeffs_nom[5]

        # Z10 does not deviate from nominal
        coeffs_field[6] = coeffs_nom[6]

        # Z11 deviation from nominal varies linearly across the field
        coeffs_field[7] = self.Z11_a * position[0] + self.Z11_b * position[1] + self.Z11_c + coeffs_nom[7]

        return coeffs_field
    
    
    def _rev_calc_coeffs_field(self, position, phasebar):

        xbar_partial = np.zeros(18)

        # Z4
        xbar_partial[0] = phasebar[0] * position[0]
        xbar_partial[1] = phasebar[0] * position[1]
        xbar_partial[2] = phasebar[0]

        # Z5
        xbar_partial[3] = phasebar[1] * position[0]
        xbar_partial[4] = phasebar[1] * position[1]
        xbar_partial[5] = phasebar[1]

        # Z6
        xbar_partial[6] = phasebar[2] * position[0] 
        xbar_partial[7] = phasebar[2] * position[1]
        xbar_partial[8] = phasebar[2]

        # Z7
        xbar_partial[9] = phasebar[3] * position[0]
        xbar_partial[10] = phasebar[3] * position[1]
        xbar_partial[11] = phasebar[3]

        # Z8
        xbar_partial[12] = phasebar[4] * position[0]
        xbar_partial[13] = phasebar[4] * position[1]
        xbar_partial[14] = phasebar[4]

        # Z11
        xbar_partial[15] = phasebar[7] * position[0]
        xbar_partial[16] = phasebar[7] * position[1]
        xbar_partial[17] = phasebar[7]

        return xbar_partial


    def fwd_field(self, x):

        self.E = 0

        self.Z4_a = x[0]
        self.Z4_b = x[1]
        self.Z4_c = x[2]

        self.Z5_a = x[3]
        self.Z5_b = x[4]
        self.Z5_c = x[5]

        self.Z6_a = x[6]
        self.Z6_b = x[7]
        self.Z6_c = x[8]

        self.Z7_a = x[9]
        self.Z7_b = x[10]
        self.Z7_c = x[11]
        
        self.Z8_a = x[12]
        self.Z8_b = x[13]
        self.Z8_c = x[14]

        self.Z11_a = x[15]
        self.Z11_b = x[16]
        self.Z11_c = x[17]

        for opt, position, coeffs_nom in zip(self.optlist, self.psf_positions, self.coeffs_field_nom):

            opt.init_phase = sum_of_2d_modes(self.modes_common, self.coeffs_common)

            opt.modes = self.modes_field
            
            coeffs_field = self._fwd_calc_coeffs_field(position, coeffs_nom)

            self.E += opt.fwd(x=coeffs_field)

        self.costs.append(self.E / len(self.optlist))

        return self.E
    
        
    def rev_field(self):

        self.xbar = np.zeros(18)

        for opt, position in zip(self.optlist, self.psf_positions):

            phasebar = opt.rev()

            self.xbar += self._rev_calc_coeffs_field(position, phasebar)

        return self.xbar

        
    def fg_field(self, x):

        f = self.fwd_field(x)

        g = self.rev_field()

        return ensure_np(f), ensure_np(g)

        
    def fwd_common(self, x):

        self.E = 0

        self.coeffs_common = np.array(x)

        for opt, position, coeffs_nom in zip(self.optlist, self.psf_positions, self.coeffs_field_nom):

            coeffs_field = self._fwd_calc_coeffs_field(position, coeffs_nom)
            
            opt.init_phase = sum_of_2d_modes(self.modes_field, coeffs_field)

            opt.modes = self.modes_common

            self.E += opt.fwd(x)

        self.costs.append(self.E / len(self.optlist))

        return self.E
    

    def rev_common(self):

        self.xbar = np.zeros(len(self.modes_common))

        for opt in self.optlist:

            self.xbar += opt.rev()

        return self.xbar
    
    
    def fg_common(self, x):

        f = self.fwd_common(x)

        g = self.rev_common()

        return ensure_np(f), ensure_np(g)
    
    
    def minimize_field(self, jac=True, method='L-BFGS-B', options={'maxls' : 10, 'ftol' : 1e-20, 'gtol' : 1e-8, 'disp' : 0, 'maxiter' : 100}):

        result = minimize(self.fg_field, x0=truenp.array([self.Z4_a, self.Z4_b, self.Z4_c, 
                                                          self.Z5_a, self.Z5_b, self.Z5_c,
                                                          self.Z6_a, self.Z6_b, self.Z6_c,
                                                          self.Z7_a, self.Z7_b, self.Z7_c,
                                                          self.Z8_a, self.Z8_b, self.Z8_c,
                                                          self.Z11_a, self.Z11_b, self.Z11_c]), jac=jac, method=method, options=options)

        return result
    

    def minimize_common(self, jac=True, method='L-BFGS-B', options={'maxls' : 10, 'ftol' : 1e-20, 'gtol' : 1e-8, 'disp' : 0, 'maxiter' : 100}):

        result = minimize(self.fg_common, x0=ensure_np(self.coeffs_common), jac=jac, method=method, options=options)

        return result

    
def BBPR(wvls, pupil, dx_pupil, psfs, dx_psf, efl, bb_parameters, 
         defocus_coeffs, error_fx='GIE', display=True):
    
    # pupil coords
    diam_pup = pupil.shape[0] * dx_pupil
    x, y = make_xy_grid(shape=pupil.shape[0], diameter=diam_pup)
    r, t = cart_to_polar(x, y)

    # zernikes to be estimated
    z_min = int(bb_parameters['general']['zernike_min'])
    z_max = int(bb_parameters['general']['zernike_max']) + 1 # add 1 due to range going to i - 1
    nms = [noll_to_nm(i) for i in range(z_min, z_max)]
    z_coeffs = list(zernike_nm_seq(nms, r, t, norm=True))
    zernikes = [z / np.max(np.abs(z)) for z in z_coeffs]

    # mask for which zernikes are field-dependent
    fd_inds = np.array(bb_parameters['general']['zernikes_field_dependent']) - bb_parameters['general']['zernike_min']
    fd_mask = np.ones(len(zernikes), dtype='bool')
    fd_mask[fd_inds] = 1

    # for storing optimization results 
    opt_opds = [np.zeros_like(pupil).get() for i in range(len(psfs))]
    opt_coeffs = [np.zeros((len(zernikes),)).get() for i in range(len(psfs))]
    opt_psfs = [np.zeros_like(psfs[0]).get() for i in range(len(psfs))]
    results = []

    for i, param in enumerate(bb_parameters):
        if param.__contains__('iteration'):

            # gain coeffs for zernikes
            gains = np.ones((len(zernikes),))
            gains[fd_mask] = bb_parameters[param]['gain_field_dependent']
            gains[~fd_mask] = bb_parameters[param]['gain_common']

            options = bb_parameters[param]['opt_options']

            # make list of ADPR classes 
            adpr_list = []
            for j in range(len(psfs)):
                adpr_list.append(ADPR(wvls=wvls.tolist(), pupil=pupil, dx_pupil=dx_pupil, psf=psfs[j], dx_psf=dx_psf, 
                                        efl=efl, modes=np.array(zernikes) * gains[:, None, None], defocus_coeff=defocus_coeffs[j], 
                                        initial_phase=np.array(opt_opds[j]), error_fx=error_fx))
                
            # individual or joint optimization
            if str(bb_parameters[param]['type']).upper() == 'JOINT':

                fdpr = FDPR(optlist=adpr_list)
                result = minimize(fdpr.fg, x0=np.zeros(len(zernikes)).get(), jac=True, method='L-BFGS-B', options=options)
                
                for j in range(len(psfs)):
                    opt_coeffs[j] += result.x * gains.get()
                    opt_opds[j] = sum_of_2d_modes(zernikes, np.array(opt_coeffs[j])).get()
                    opt_psfs[j] = fdpr.optlist[j].I.get()

            elif str(bb_parameters[param]['type']).upper() == 'INDIVIDUAL':

                for j in range(len(psfs)):
                    result = minimize(adpr_list[j].fg, x0=np.zeros(len(zernikes)).get(), jac=True, method='L-BFGS-B', options=options)
                    opt_coeffs[j] += result.x * gains.get()
                    opt_opds[j] = sum_of_2d_modes(zernikes, np.array(opt_coeffs[j])).get()
                    opt_psfs[j] = fdpr.optlist[j].I.get()
            
            # store results
            results.append({'coeffs' : opt_coeffs,
                            'opds'   : opt_opds,
                            'psfs'   : opt_psfs})

            # if desired, display costs
            if display:
                for j, opt in enumerate(adpr_list):
                    plt.plot(np.asarray(opt.costs).get(), label=f'Defocus : {defocus_coeffs[j]:0.2f}$\lambda$', alpha=0.4)
                plt.title(f'BBPR Iteration {i:.0f}')
                plt.ylabel('Error Function')
                plt.xlabel('Optimizer Iteration')
                plt.legend(loc='upper right')
                plt.yscale('log')
                plt.show()

    return results



