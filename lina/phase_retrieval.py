from prysm.mathops import np

from prysm.propagation import (
    focus_fixed_sampling,
    focus_fixed_sampling_backprop
)

from prysm.coordinates import (
    make_xy_grid,
    cart_to_polar
)

from prysm.polynomials import (
    hopkins
)

"""Largely taken from poi.phase_retrieval, with modifications to support spectral diversity"""
class ADPhaseRetireval:
    def __init__(self, amp, amp_dx, efl, wvls, basis, target, img_dx, defocus_waves=0, 
                 initial_phase=None, error_function='MSE'):
        if initial_phase is None:
            phs = np.zeros(amp.shape, dtype=float)
        else:
            phs = initial_phase

        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.amp_dx = amp_dx
        self.epd = amp.shape[0] * amp_dx
        self.efl = efl
        self.wvls = wvls
        self.basis = basis
        self.img_dx = img_dx
        self.D = target
        self.init_phs = phs
        self.phs = phs
        self.zonal = False
        self.defocus = defocus_waves
        self.error_fx = error_function.upper()
        
        # check for valid error function
        if self.error_fx != 'MSE' and self.error_fx != 'GIE':
            self.error_fx = 'MSE'
            print("INVALID ERROR FUNCTION PROVIDED, DEFAULTING TO MSE")

        # configure the defocus polynomial
        x, y = make_xy_grid(amp.shape[0], diameter=self.epd)
        r, t = cart_to_polar(x, y)
        r_z = r / (self.epd / 2)
        self.defocus_polynomial = hopkins(0, 2, 0, r_z, t, 0)
        self.defocus_aberration = 2 * np.pi * self.defocus_polynomial * self.defocus * self.amp
        self.cost = []

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        if not self.zonal:
            if len(x) == 1:
                phs = self.init_phs + np.asarray(self.basis) * np.asarray(x)
            else:
                phs = self.init_phs + np.tensordot(np.asarray(self.basis), 
                                                   np.asarray(x), axes=(0,0))
        else:
            phs = self.init_phs + np.zeros(self.amp.shape, dtype=float)
            phs[self.amp_select] = x
        
        I = 0

        for wvl in self.wvls:

            W = (2 * np.pi / wvl) * phs

            # TODO: Check if this is a minus sign instead
            W -= self.defocus_aberration
            g = self.amp * np.exp(1j * W)
            G = focus_fixed_sampling(
                wavefunction=g,
                input_dx=self.amp_dx,
                prop_dist = self.efl,
                wavelength=wvl,
                output_dx=self.img_dx,
                output_samples=self.D.shape,
                shift=(0, 0),
                method='mdft')
            I += np.abs(G)**2 / len(self.wvls)
        
        if self.error_fx == 'MSE':
            E = mse(I, self.D)
        elif self.error_fx == 'GIE':
            E = gie(I, self.D)

        self.phs = phs
        self.W = W
        self.g = g
        self.G = G
        self.I = I
        self.E = E
        return

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        self.update(x)

        Wbar = 0

        for wvl in self.wvls:
            if self.error_fx == 'MSE':
                Ibar = grad_mse(self.I, self.D) / len(self.wvls)
            elif self.error_fx == 'GIE':
                Ibar = grad_gie(self.I, self.D) / len(self.wvls)
            Gbar = 2 * Ibar * self.G
            gbar = focus_fixed_sampling_backprop(
                wavefunction=Gbar,
                input_dx=self.amp_dx,
                prop_dist = self.efl,
                wavelength=wvl,
                output_dx=self.img_dx,
                output_samples=self.phs.shape,
                shift=(0, 0),
                method='mdft')

            Wbar += 2 * np.pi / wvl * np.imag(gbar * np.conj(self.g))
            
        if not self.zonal:
            abar = np.tensordot(self.basis, Wbar)

        self.Ibar = Ibar
        self.Gbar = Gbar
        self.gbar = gbar
        self.Wbar = Wbar

        if not self.zonal:
            self.abar = abar
            return self.abar
        else:
            return self.Wbar[self.amp_select]

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        self.cost.append(f)
        return f.get(), g.get()
    

class ParallelADPhaseRetrieval:

    def __init__(self, optlist):

        self.optlist = optlist
        self.f = 0
        self.g = 0
        self.cost = []

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
        
        self.cost.append(self.f)

        return self.f, self.g


def mse(I, D):
    return np.sum((I - D) ** 2)


def grad_mse(I, D):
    return 2 * (I - D)


def gie(I, D):
    t1 = np.sum(I * D) ** 2
    t2 = np.sum(D ** 2) 
    t3 = np.sum(I ** 2)
    return 1 - t1 / (t2 * t3)


def grad_gie(I, D):
    t1 = np.sum(I * D)
    t2 = np.sum(D ** 2)
    t3 = np.sum(I ** 2)
    return 2 * t1 / (t2 * t3 ** 2) * (I * t1 - D * t3)