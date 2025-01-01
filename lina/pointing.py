import astropy.units as u
import matplotlib.pyplot as plt

import numpy as np


def tilt_noise(tel,
               mag,
                t_exp="1second",
                inst_name="IMAGER",
                N=4,
                dx_aperture=0.25,
               phot_e_rate=None,
               return_SNR=False):
    """
    Calculates the tilt noise for an observatory object class __tel__ from first principles 
    as described in Lindgren Lindegren 2013 https://doi.org/10.1007/978-1-4614-7804-1_16 
    adapted from code https://zenodo.org/record/4499318 
    released to accompany Douglas et al (http://arxiv.org/abs/2112.12835) 

    Expected Parameters:
    mag (float): Magnitude of the observed object. 
    phot_e_rate (Quantity, optional): Photoelectron rate. If not provided, it will be calculated based on a Johnson V-magnitude.
    t_exp (Quantity): Exposure time in seconds.

    N (int): Number of pixels in the centroiding aperture. 
        Defaults to 4, the minimum number of pixels for centroiding (e.g. a quadcell).
    inst_name= attribute of telescope class of the instrument to calculate from, defaults to "IMAGER".
    tel.D (Quantity): Diameter of the telescope aperture.
    tel.dx_aperture (float): delta-x, the standard deviation scaling to apply. 
        See Lindegren 2013 eq. 16.1, https://doi.org/10.1007/978-1-4614-7804-1_16)
        Defaults to dx_aperture = 1/4 for a circular aperture of diameter D,
         dx_aperture = L/sqrt(12) for a rectangular aperture of length L,
         dx_aperture = B/2 for a two-element interferometer baseline  of B.
    inst.wavel (Quantity): Central wavelength of the observed light in nanometers.
    inst.RN (Quantity): Read noise in electrons, assumes one read per measurement.
    inst.D_i (Quantity): Dark current in electrons per second.
    inst.E (Quantity): Quantum efficiency in electrons per photon.
    inst.bw (Quantity): Bandwidth in nanometers.

    return_SNR (bool): If True, the signal-to-noise ratio (SNR) will be returned instead of the tilt noise.

    Returns:
    Quantity: Tilt noise in arcseconds, unless return_SNR is True, in which case the SNR is returned.
    """
    #todo: add logging debug statements
    inst = getattr(tel, inst_name)
    D = tel.telescope.D_aperture
    if phot_e_rate is None:
        F0=948*u.photon/u.second/u.Angstrom/u.cm**2
        photons=F0*10**(-mag/2.5)
        phot_e_rate=(D/2)**2*np.pi*inst.bw*inst.QE*photons

    SNR=(phot_e_rate*t_exp).to(u.electron).value/np.sqrt((phot_e_rate*t_exp).to(u.electron).value
                                                    +((inst.RN*N)**2).value
                                                   +(N*inst.D_i*t_exp).to(u.electron).value)
    err= ((1/(4*np.pi)*inst.wavel/(dx_aperture * D)/SNR)*u.radian).to(u.arcsec)
    
    if return_SNR:
        return err, SNR
    else:
        return err


def tilt_noise_photons(photons,tel):
    #print((e_rate*t_exp).decompose())
    err= ((1/(4*np.pi)*wavel/(dx_aperture * D)/np.sqrt(photons))*u.radian).to(u.arcsec)

    return err  
