from .math_module import xp, xcipy, ensure_np_array
from lina import utils

import numpy as np
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path
import copy

def fft(arr):
    return xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(arr)))
    # return xp.fft.ifftshift(xp.fft.fft2(arr))

def ifft(arr):
    return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(arr)))
    # return xp.fft.fftshift(xp.fft.ifft2(arr))

def ang_spec(wavefront, wavelength, distance, pixelscale):
    n = wavefront.shape[0]

    delkx = 2*np.pi/(n*pixelscale)
    kxy = (xp.linspace(-n/2, n/2-1, n) + 1/2)*delkx
    k = 2*np.pi/wavelength
    kx, ky = xp.meshgrid(kxy,kxy)

    wf_as = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(wavefront)))
    
    kz = xp.sqrt(k**2 - kx**2 - ky**2 + 0j)
    tf = xp.exp(1j*kz*distance)

    prop_wf = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(wf_as * tf)))
    kz = 0.0
    tf = 0.0

    return prop_wf

def make_mft_forward_matrices(
        N, 
        npix, 
        npsf, 
        psf_pixelscale_lamD, 
        convention='-', 
        pp_centering='odd', 
        fp_centering='odd',
    ):
    dx = 1.0 / npix
    if pp_centering=='even':
        Xs = (xp.arange(N, dtype=float) - (N / 2) + 1/2) * dx
    elif pp_centering=='odd':
        Xs = (xp.arange(N, dtype=float) - (N / 2)) * dx

    du = psf_pixelscale_lamD
    if fp_centering=='odd':
        Us = (xp.arange(npsf, dtype=float) - npsf / 2) * du
    elif fp_centering=='even':
        Us = (xp.arange(npsf, dtype=float) - npsf / 2 + 1/2) * du

    xu = xp.outer(Us, Xs)
    vy = xp.outer(Xs, Us)

    if convention=='-':
        My = xp.exp(-1j*2*np.pi*vy) 
        Mx = xp.exp(-1j*2*np.pi*xu)
    else:
        My = xp.exp(1j*2*np.pi*vy) 
        Mx = xp.exp(1j*2*np.pi*xu)

    norm_coeff = psf_pixelscale_lamD/npix

    return Mx, My, norm_coeff

def mft_forward(
        wavefront, 
        npix, 
        npsf, 
        psf_pixelscale_lamD, 
        convention='-', 
        pp_centering='odd', 
        fp_centering='odd',
    ):
    N = wavefront.shape[0]
    dx = 1.0 / npix
    if pp_centering=='even':
        Xs = (xp.arange(N, dtype=float) - (N / 2) + 1/2) * dx
    elif pp_centering=='odd':
        Xs = (xp.arange(N, dtype=float) - (N / 2)) * dx

    du = psf_pixelscale_lamD
    if fp_centering=='odd':
        Us = (xp.arange(npsf, dtype=float) - npsf / 2) * du
    elif fp_centering=='even':
        Us = (xp.arange(npsf, dtype=float) - npsf / 2 + 1/2) * du

    xu = xp.outer(Us, Xs)
    vy = xp.outer(Xs, Us)

    if convention=='-':
        My = xp.exp(-1j*2*np.pi*vy) 
        Mx = xp.exp(-1j*2*np.pi*xu)
    else:
        My = xp.exp(1j*2*np.pi*vy) 
        Mx = xp.exp(1j*2*np.pi*xu)

    norm_coeff = psf_pixelscale_lamD/npix

    return Mx@wavefront@My * norm_coeff

def make_mft_reverse_matrices(
        npsf, 
        psf_pixelscale_lamD, 
        npix, 
        N, 
        convention='+', 
        pp_centering='odd', 
        fp_centering='odd',
    ):
    du = psf_pixelscale_lamD
    if fp_centering=='odd':
        Us = (xp.arange(npsf, dtype=float) - npsf / 2) * du
    elif fp_centering=='even':
        Us = (xp.arange(npsf, dtype=float) - npsf / 2 + 1/2) * du

    dx = 1.0 / npix
    if pp_centering=='even':
        Xs = (xp.arange(N, dtype=float) - (N / 2) + 1/2) * dx
    elif pp_centering=='odd':
        Xs = (xp.arange(N, dtype=float) - (N / 2)) * dx

    ux = xp.outer(Xs, Us)
    yv = xp.outer(Us, Xs)

    if convention=='+':
        My = xp.exp(1j*2*np.pi*yv) 
        Mx = xp.exp(1j*2*np.pi*ux)
    else:
        My = xp.exp(-1j*2*np.pi*yv)
        Mx = xp.exp(-1j*2*np.pi*ux)

    norm_coeff = psf_pixelscale_lamD/npix 

    return Mx, My, norm_coeff

def mft_reverse(
        fpwf, 
        psf_pixelscale_lamD,
        npix, 
        N, 
        convention='+', 
        pp_centering='odd', 
        fp_centering='odd',
    ):
    npsf = fpwf.shape[0]
    du = psf_pixelscale_lamD
    if fp_centering=='odd':
        Us = (xp.arange(npsf, dtype=float) - npsf / 2) * du
    elif fp_centering=='even':
        Us = (xp.arange(npsf, dtype=float) - npsf / 2 + 1/2) * du

    dx = 1.0 / npix
    if pp_centering=='even':
        Xs = (xp.arange(N, dtype=float) - (N / 2) + 1/2) * dx
    elif pp_centering=='odd':
        Xs = (xp.arange(N, dtype=float) - (N / 2)) * dx

    ux = xp.outer(Xs, Us)
    yv = xp.outer(Us, Xs)

    if convention=='+':
        My = xp.exp(1j*2*np.pi*yv) 
        Mx = xp.exp(1j*2*np.pi*ux)
    else:
        My = xp.exp(-1j*2*np.pi*yv)
        Mx = xp.exp(-1j*2*np.pi*ux)

    norm_coeff = psf_pixelscale_lamD/npix 

    return Mx@fpwf@My * norm_coeff

def get_scaled_coords(N, scale, center=True, shift=True):
    if center:
        cen = (N-1)/2.0
    else:
        cen = 0
        
    if shift:
        shiftfunc = xp.fft.fftshift
    else:
        shiftfunc = lambda x: x
    cy, cx = (shiftfunc(xp.indices((N,N))) - cen) * scale
    r = xp.sqrt(cy**2 + cx**2)
    return [cy, cx, r]

def get_fresnel_TF(dz, N, wavelength, fnum):
    '''
    Get the Fresnel transfer function for a shift dz from focus
    '''
    df = 1.0 / (N * wavelength * fnum)
    rp = get_scaled_coords(N,df, shift=False)[-1]
    return xp.exp(-1j*np.pi*dz*wavelength*(rp**2))

def make_vortex_phase_mask(
        npix, charge=6, 
        grid='odd', 
        singularity=None, 
        focal_length=500*u.mm, 
        pupil_diam=9.5*u.mm, 
        wavelength=650*u.nm,
    ):
    
    if grid=='odd':
        x = xp.linspace(-npix//2, npix//2-1, npix)
    elif grid=='even':
        x = xp.linspace(-npix//2, npix//2-1, npix) + 1/2
    x,y = xp.meshgrid(x,x)
    th = xp.arctan2(y,x)

    phasor = xp.exp(1j*charge*th)
    
    if singularity is not None:
        r = xp.sqrt((x-1/2)**2 + (y-1/2)**2)
        mask = r>(singularity*pupil_diam/(focal_length*wavelength)).decompose().value
        phasor *= mask
    
    return phasor


