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
                 defocus_coeff, initial_phase=None, error_fx='GIE'):
        
        # ADPR parameters
        self.wvls = wvls
        self.pup = pupil
        self.dx_pup = dx_pupil
        self.D_pup = pupil.shape[0] * dx_pupil
        self.efl = efl
        self.modes = modes
        self.psf = psf
        self.dx_psf = dx_psf
        self.error_fx = error_fx.upper()
        self.costs = []

        if initial_phase is None:
            self.init_phase = np.zeros(pupil.shape, dtype=float)
        else:
            self.init_phase = initial_phase

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

        self.phase = self.init_phase + sum_of_2d_modes(self.modes, np.array(x))

        self.Ws = []
        self.gs = []
        self.Gs = []
        self.I = 0

        for wvl in self.wvls:

            W = (2 * np.pi / wvl) * (self.phase - self.defocus) / 1e3
            self.Ws.append(W)

            g = self.pup * np.exp(1j * W)
            self.gs.append(g)

            G = focus_fixed_sampling(wavefunction=g, input_dx=self.dx_pup, prop_dist=self.efl, wavelength=wvl,
                                     output_dx=self.dx_psf, output_samples=self.psf.shape[0], shift=(0, 0), method='mdft')
            self.Gs.append(G)

            self.I += np.abs(G) ** 2 / len(self.wvls)

        self.E = self.err(self.I, self.psf)

        self.costs.append(self.E)

        return self.E
    
    def rev(self):

        self.Ibars = []
        self.Gbars = []
        self.gbars = []
        self.Wbar = 0

        for wvl, G, g in zip(self.wvls, self.Gs, self.gs):

            Ibar = self.grad_err(self.I, self.psf) / len(self.wvls)
            self.Ibars.append(Ibar)

            Gbar = 2 * Ibar * G 
            self.Gbars.append(Gbar)

            gbar = focus_fixed_sampling_backprop(wavefunction=Gbar, input_dx=self.dx_pup, prop_dist=self.efl, wavelength=wvl,
                                                 output_dx=self.dx_psf, output_samples=self.pup.shape[0], shift=(0, 0), method='mdft')
            self.gbars.append(gbar)

            self.Wbar += (2 * np.pi / wvl) * np.imag(gbar * np.conj(g)) / 1e3

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

