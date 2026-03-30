from .math_module import xp, xcipy, ensure_np_array
from lina import utils
from lina.utils import create_zernike_modes

import numpy as np
import scipy
import astropy.units as u
import copy
import matplotlib.pyplot as plt

import poppy

def noll_index_to_mn(j):
    # j is the Noll index of the Zernike term. 
    # Note that Noll Zernikes start with j=1, which is the piston term. 
    n = int(np.floor( (np.sqrt(8*(j-1) + 1) - 1)/2 ))
    m = int( (-1)**j * ( n%2 + 2*np.floor( (j - n*(n+1)/2 - 1 + (n+1)%2 )/2 ) ))

    return m,n

def mn_to_noll_index(m,n):
    assert isinstance(m, int) and isinstance(n, int)
    if n%4<=1:
        if m>0:
            j = n*(n+1)/2 + abs(m) + 0
        elif m<=0:
            j = n*(n+1)/2 + abs(m) + 1
    elif n%4>1:
        if m<0:
            j = n*(n+1)/2 + abs(m) + 0
        elif m>=0:
            j = n*(n+1)/2 + abs(m) + 1
    return int(j)

def fringe_index_to_mn(j):
    g = np.ceil(np.sqrt(j) - 1)
    n = g - 1 + np.ceil((j-g**2)/2)
    m = (-1)**( (j-g**2)%2 + 1 ) * (2*g - n)
    return int(m), int(n)

def mn_to_fringe_index(m,n):
    # j = (1 + (n + abs(m))/2)**2 - 2*abs(m) + 0 if m<=0 else 1
    assert isinstance(m, int) and isinstance(n, int)
    if 0<=m:
        j = (1 + (n + abs(m))/2)**2 - 2*abs(m) + 0
    elif 0>m:
        j = (1 + (n + abs(m))/2)**2 - 2*abs(m) + 1
    return int(j)

def generate_opd(
        npix=1000, 
        oversample=1, 
        wavelength=500*u.nm,
        slope=2.5, 
        seed=1234, 
        rms=10, 
        remove_modes=3, # defaults to removing piston, tip, and tilt
    ):
    diam = 10*u.mm
    wf = poppy.FresnelWavefront(beam_radius=diam/2, npix=npix, oversample=oversample, wavelength=wavelength)
    wfe_opd = poppy.StatisticalPSDWFE(index=slope, wfe=rms, radius=diam/2, seed=seed).get_opd(wf)
    circ = poppy.CircularAperture(radius=diam/2).get_transmission(wf)
    bmask = circ>0
    
    wfe_opd = xp.asarray(wfe_opd)

    if remove_modes>0:
        Zs = poppy.zernike.arbitrary_basis(circ, nterms=remove_modes, outside=0)
        Zc_opd = utils.lstsq(Zs, wfe_opd)
        for i in range(remove_modes):
            wfe_opd -= Zc_opd[i] * Zs[i]
    wfe_rms = xp.sqrt( xp.mean( xp.square( wfe_opd[bmask] )))
    wfe_opd *= rms.to_value(u.m)/wfe_rms

    return wfe_opd*circ

def generate_wfe(
        npix=1000, 
        oversample=1, 
        wavelength=500*u.nm,
        opd_index=2.5, 
        amp_index=2.5, 
        opd_seed=1234, 
        amp_seed=12345,
        opd_rms=10*u.nm, 
        amp_rms=0.05,
        remove_opd_modes=3,
        remove_amp_modes=3, # defaults to removing piston, tip, and tilt
    ):
    diam = 10*u.mm
    opd_seed = xp.uint64(opd_seed)
    amp_seed = xp.uint64(amp_seed)
    wf = poppy.FresnelWavefront(beam_radius=diam/2, npix=npix, oversample=oversample, wavelength=wavelength)
    wfe_amp = poppy.StatisticalPSDWFE(index=amp_index, wfe=amp_rms*u.nm, radius=diam/2, seed=amp_seed).get_opd(wf)
    wfe_opd = poppy.StatisticalPSDWFE(index=opd_index, wfe=opd_rms, radius=diam/2, seed=opd_seed).get_opd(wf)
    circ = poppy.CircularAperture(radius=diam/2).get_transmission(wf)
    bmask = circ>0
    
    wfe_amp = xp.asarray(wfe_amp)
    wfe_opd = xp.asarray(wfe_opd)

    if remove_amp_modes>0:
        Zs = poppy.zernike.arbitrary_basis(circ, nterms=remove_amp_modes, outside=0)
        Zc_amp = utils.lstsq(Zs, wfe_amp)
        for i in range(remove_amp_modes):
            wfe_amp -= Zc_amp[i] * Zs[i]
    wfe_amp = wfe_amp*1e9 + 1

    if remove_opd_modes>0:
        Zs = poppy.zernike.arbitrary_basis(circ, nterms=remove_opd_modes, outside=0)
        Zc_opd = utils.lstsq(Zs, wfe_opd)
        for i in range(remove_opd_modes):
            wfe_opd -= Zc_opd[i] * Zs[i]
    wfe_rms = xp.sqrt( xp.mean( xp.square( wfe_opd[bmask] )))
    wfe_opd *= opd_rms.to_value(u.m)/wfe_rms

    return wfe_amp*circ, wfe_opd*circ

def generate_freqs(
        Nf=2**18+1, 
        f_min=0, 
        f_max=1000,
        return_np=False,
    ):
    """_summary_

    Parameters
    ----------
    Nf : Number of samples for the frequency range, optional
         must be supplied as a power of 2 plus 1, by default 2**18+1
    f_min : _type_, optional
        _description_, by default 0*u.Hz
    f_max : _type_, optional
        _description_, by default 100*u.Hz

    Returns
    -------
    _type_
        _description_
    """
    # if bin(Nf-1).count('1')!=1: 
        # raise ValueError('Must supply number of samples to be a power of 2 plus 1. ')
    del_f = (f_max - f_min)/Nf
    # freqs = xp.arange(f_min, f_max, del_f)
    freqs = xp.linspace(f_min, f_max, Nf)
    Nt = 2*(Nf-1)
    del_t = 1/(2*f_max)
    times = xp.linspace(0, (Nt-1)*del_t, Nt)
    if return_np:
        return ensure_np_array(freqs), ensure_np_array(times)
    return freqs, times

def roll_psd(
        freqs, 
        beta, 
        f_roll, 
        alpha,
        normalized=True,
        verbose=True,
    ):
    if normalized:
        psd = beta**2 / (1+freqs/f_roll)**alpha * (alpha - 1)/f_roll # the last factor is to make sure the RMS of the PSD is normalized correctly
    else:
        psd = beta**2 / (1+freqs/f_roll)**alpha

    if verbose: 
        psd_rms = np.sqrt(scipy.integrate.simpson(ensure_np_array(psd), x=ensure_np_array(freqs)))
        print(f'\tRMS of generated knee PSD: {psd_rms:.3e}')

    return psd

def generate_time_series(
        psd, 
        f_max, 
        rms=None,  
        seed=123,
        return_times=False,
        return_np=False,
    ):
    Nfreq_samps = len(psd)
    Ntime_samps = 2 * (Nfreq_samps - 1)
    del_time = 1/(2*f_max)
    times = xp.linspace(0, (Ntime_samps-1)*del_time, Ntime_samps)

    P_fft_one_sided = copy.copy(psd)

    # Because P includes both DC and Nyquist (N/2+1), P_fft must have 2*(N_P-1) elements
    P_fft_one_sided[0] = 2 * P_fft_one_sided[0]
    P_fft_one_sided[-1] = 2 * P_fft_one_sided[-1]

    P_fft_new = xp.zeros((Ntime_samps,), dtype=complex)
    P_fft_new[0:int(Ntime_samps/2)+1] = P_fft_one_sided
    P_fft_new[int(Ntime_samps/2)+1:] = P_fft_one_sided[-2:0:-1]

    # Take the square root to get the amplitude of the power spectrum
    amplitude_spectrum = xp.sqrt(P_fft_new)

    # Create random phases for all FFT terms other than DC and Nyquist
    xp.random.seed(xp.uint64(seed))
    phases = xp.random.uniform(0, 2*np.pi, (int(Ntime_samps/2),))

    # Ensure X_new has complex conjugate symmetry
    amplitude_spectrum[1:int(Ntime_samps/2)+1] = amplitude_spectrum[1:int(Ntime_samps/2)+1] * np.exp(2j*phases)
    amplitude_spectrum[int(Ntime_samps/2):] = amplitude_spectrum[int(Ntime_samps/2):] * np.exp(-2j*phases[::-1])
    amplitude_spectrum = amplitude_spectrum * np.sqrt(Ntime_samps) / np.sqrt(2)

    # This is the new time series with a given PSD
    time_series = xcipy.fft.ifft(amplitude_spectrum)
    # print(x_new.real, x_new.imag)
    time_series = time_series.real

    if rms is not None: 
        time_series *= rms/np.sqrt(np.mean(np.square(time_series)))
    
    if return_np:
        time_series = ensure_np_array(time_series)
        times = ensure_np_array(times)

    if return_times:
        return time_series, times
    
    return time_series

def compute_cumulative_psd(freqs, psd):
    cumulative_psd = []
    for i in range(1,len(freqs)):
        psd_domain = freqs[0:i]
        psd_range = psd[0:i]
        psd_integral = scipy.integrate.simpson(psd_range, x=psd_domain)
        cumulative_psd.append(psd_integral)

    cumulative_psd = np.sqrt(np.array(cumulative_psd))
    return cumulative_psd, freqs[1:]


def plot_psd(freqs, psd, plot_integral=False):
    plt.plot(freqs, psd)
    plt.title(f"temporal PSD")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.xlabel(freqs.unit)
    plt.show()

    if plot_integral:
        Nints = 2000
        Nf = len(freqs)
        del_f = freqs[1]-freqs[0]
        int_freqs = []
        psd_int = []
        for i in range(Nints):
            i_psd = int(np.round(Nf/(Nints-i)))
            int_freqs.append(freqs[i_psd-1].to_value(u.Hz))
            psd_int.append(np.trapz(psd[:i_psd])*del_f.to_value(u.Hz))

        plt.plot(int_freqs, psd_int)
        plt.title(f"Integral of temporal PSD over frequency range")
        plt.yscale("log")
        plt.xscale("log")
        plt.grid()
        plt.xlabel(freqs.unit)
        plt.show()

def plot_time_series(times, coeff, name='Coefficients', xlims=None, ylims=None):
    times = ensure_np_array(times)
    coeff = ensure_np_array(coeff)
    c_rms = np.sqrt(np.mean(np.square(coeff)))
    plt.plot(times, coeff)
    plt.title(f'Time series of {name}, RMS = {c_rms:.3e}')
    plt.ylabel(f'{name} Amplitudes')
    plt.grid()
    plt.xlim(xlims)
    if ylims is None: 
        plt.ylim([-2*c_rms, 2*c_rms])
    else:
        plt.ylim(ylims)
    plt.xlabel('Seconds')
    plt.show()

def plot_psd_and_time_series(
        freqs, psd, times, coeff,
        psd_name='PSD', 
        psd_xlims=None, psd_ylims=None,
        coeff_name='Coefficients', 
        coeff_xlims=None, coeff_ylims=None,
        coeff_xticks=None,
        figsize=(16,9),
        dpi=125,
    ):
    freqs = ensure_np_array(freqs)
    psd = ensure_np_array(psd)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=dpi)
    
    ax[0].plot(freqs,psd)
    ax[0].set_title(f'Temporal PSD {psd_name}')
    ax[0].set_yscale("log")
    ax[0].set_xscale("log")
    ax[0].set_ylabel(f'PSD Amplitude')
    ax[0].grid()
    ax[0].set_xlim(psd_xlims)
    ax[0].set_ylim(psd_ylims)
    ax[0].set_xlabel('Hz')

    times = ensure_np_array(times)
    coeff = ensure_np_array(coeff)
    c_rms = np.sqrt(np.mean(np.square(coeff)))
    ax[1].plot(times, coeff)
    ax[1].set_title(f'Time series of {coeff_name}, RMS = {c_rms:.3e}')
    ax[1].set_ylabel(f'{coeff_name} Amplitudes')
    if coeff_xticks is not None: ax[1].set_xticks(coeff_xticks)
    ax[1].grid()
    ax[1].set_xlim(coeff_xlims)
    ax[1].set_ylim(coeff_ylims)
    ax[1].set_xlabel('Seconds')

    plt.show()


