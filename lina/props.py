from .math_module import xp, xcipy, ensure_np_array
from lina import utils

import numpy as np
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path
import copy

def fft(arr, sync=False):
    """
    Compute the 2D Fourier Transform of an array with proper FFT shifts applied for the DC component. 

    Args:
        arr (ndarray): 
            2D array to be Fourier Transformed.

    Returns:
        ndarray:
            The Fourier Transform of the input array.  
    """
    out = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(arr)))
    
    if sync:
        xp.cuda.Device().synchronize()

    return out
    # return xp.fft.ifftshift(xp.fft.fft2(arr))

def ifft(arr):
    """
    Compute the 2D Fourier Transform of an array with proper FFT shifts applied for the DC component. 

    Args:
        arr (ndarray): 
            2D array to be Fourier Transformed.

    Returns:
        ndarray:
            The Fourier Transform of the input array.  
    """
    return xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(arr)))
    # return xp.fft.fftshift(xp.fft.ifft2(arr))

def ang_spec(wavefront, wavelength, distance, pixelscale):
    """
    Prpoagates a wavefront a given distance using the angular spectrum propagation method. 
    The resulting wavefront will have the same pixelscale as the input wavefront. 

    Args:
        wavefront (ndarray): 
            A 2D array representing the wavefront to be propagated. 
        wavelength (float): 
            The wavelength of the wavefront in units of meters. 
        distance (float): 
            The distance to propagate in units of meters. 
        pixelscale (float): 
            The pixelscale of the current wavefront in units of meters/pixel. 

    Returns:
        ndarray: 
            The 2D array for the propagated wavefront. 
    """
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
    """
    Generate the matrices required to perform a forward Matrix Fourier Transform. 

    Args:
        N (int): 
            The number of pixels across the array that will be Fourier Transformed
        npix (int): 
            The number of pixels across pupil/beam in the array 
            that will be Fourier Transformed. 
        npsf (int): 
            The number of pixels across the focal plane the Fourier Transform will compute. 
        psf_pixelscale_lamD (float): 
            The desired pixelscale of the focal plane the Fourier Transform will compute
            in units of lambda/D. 
        convention (str, optional): 
            The sign convention to use for the Fourier Transform. Defaults to '-'. 
        pp_centering (str, optional): 
            The centering of the input wavefront. Odd means centered on a single pixel,
            even means ceneterd at the quadrant of four pixels. Defaults to 'odd'.
        fp_centering (str, optional): 
            The desired centering of the focal plane wavefront. Odd means centered on a single pixel,
            even means ceneterd at the quadrant of four pixels. Defaults to 'odd'.

    Returns:
        tuple:
            Returns two ndarrays and a float for the normalization coefficient for the MFT. 
    """


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
        sync=False,
    ):
    """
    Generate the matrices required to perform a forward Matrix Fourier Transform. 

    Args:
        wavefront (ndarray): 
            The input wavefront that will be Fourier Transformed
        npix (int): 
            The number of pixels across pupil/beam in the array 
            that will be Fourier Transformed. 
        npsf (int): 
            The number of pixels across the focal plane the Fourier Transform will compute. 
        psf_pixelscale_lamD (float): 
            The desired pixelscale of the focal plane the Fourier Transform will compute
            in units of lambda/D. 
        convention (str, optional): 
            The sign convention to use for the Fourier Transform. Defaults to '-'. 
        pp_centering (str, optional): 
            The centering of the input wavefront. Odd means centered on a single pixel,
            even means ceneterd at the quadrant of four pixels. Defaults to 'odd'.
        fp_centering (str, optional): 
            The desired centering of the focal plane wavefront. Odd means centered on a single pixel,
            even means ceneterd at the quadrant of four pixels. Defaults to 'odd'.

    Returns:
        ndarray:
            The Fourier Transform of the input wavefront. 
            
    """
    N = wavefront.shape[0]

    Mx, My, norm_coeff = make_mft_forward_matrices(
        N, 
        npix, 
        npsf, 
        psf_pixelscale_lamD, 
        convention=convention, 
        pp_centering=pp_centering, 
        fp_centering=fp_centering,
    )

    out = Mx@wavefront@My * norm_coeff

    if sync: 
        xp.cuda.Device().synchronize()

    return out

def make_mft_reverse_matrices(
        npsf, 
        psf_pixelscale_lamD, 
        npix, 
        N, 
        convention='+', 
        pp_centering='odd', 
        fp_centering='odd',
    ):
    """
    Generate the matrices required to perform a reverse Matrix Fourier Transform. 

    Args:
        npsf (int): 
            The number of pixels across the focal plane array that will be Fourier Transformed
        psf_pixelscale_lamD (float): 
            The pixelscale of the focal plane array that will be the Fourier Transformed
            in units of lambda/D. 
        npix (int): 
            The desired number of pixels across pupil plane wavefront to be computed. 
        N (int): 
            The total number of pixels across the pupil plane wavefront to be computed. 
        convention (str, optional): 
            The sign convention to use for the Fourier Transform. Defaults to '+'. 
        pp_centering (str, optional): 
            The desired centering of the pupil plane wavefront. Odd means centered on a single pixel,
            even means ceneterd at the quadrant of four pixels. Defaults to 'odd'.
        fp_centering (str, optional): 
            The centering of the focal plane wavefront. Odd means centered on a single pixel,
            even means ceneterd at the quadrant of four pixels. Defaults to 'odd'.

    Returns:
        tuple:
            Returns two ndarrays and a float for the normalization coefficient for the MFT. 
    """

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
    """
    Generate the matrices required to perform a reverse Matrix Fourier Transform. 

    Args:
        fpwf (ndarray): 
            The focal plane wavefront to compute the Fourier Transform of. 
        psf_pixelscale_lamD (float): 
            The pixelscale of the focal plane array that will be the Fourier Transformed
            in units of lambda/D. 
        npix (int): 
            The desired number of pixels across pupil plane wavefront to be computed. 
        N (int): 
            The total number of pixels across the pupil plane wavefront to be computed. 
        convention (str, optional): 
            The sign convention to use for the Fourier Transform. Defaults to '+'. 
        pp_centering (str, optional): 
            The desired centering of the pupil plane wavefront. Odd means centered on a single pixel,
            even means ceneterd at the quadrant of four pixels. Defaults to 'odd'.
        fp_centering (str, optional): 
            The centering of the focal plane wavefront. Odd means centered on a single pixel,
            even means ceneterd at the quadrant of four pixels. Defaults to 'odd'.

    Returns:
        ndarray:
            The Fourier Transform of the input wavefront. 
    """
    npsf = fpwf.shape[0]
    
    Mx, My, norm_coeff = make_mft_reverse_matrices(
        npsf, 
        psf_pixelscale_lamD, 
        npix, 
        N, 
        convention=convention, 
        pp_centering=pp_centering, 
        fp_centering=fp_centering,
    )

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
    """
    Get the Fresnel transfer function for a shift dz from focus. This transfer function
    is a form of defocus that will be applied to the pupil plane wavefront to compute a 
    defocused focal plane wavefront. 

    Args:
        dz (float): 
            The shift from nominal focus in units of meters. 
        N (int): 
            The number of pixels across the array to compute the transfer function for. 
        wavelength (float):
            The wavelength to compute the transfer function for in units of meters. 
        fnum (float): 
            The F-number of the beam to the focal plane. 

    Returns:
        ndarray: 
            The transfer function as a complex phasor. 
    """

    df = 1.0 / (N * wavelength * fnum)
    rp = get_scaled_coords(N,df, shift=False)[-1]
    return xp.exp(-1j*np.pi*dz*wavelength*(rp**2))

def make_vortex_phase_mask(
        npix, 
        charge=6, 
        grid='odd'
    ):
    """
    Compute the phasor for a vortex phase mask. 

    Args:
        npix (int): 
            The number of pixels across the phase mask array. 
        charge (int, optional): 
            The number of 2pi phase perturbations along a circular path 
            centered on the vortex phase mask. Defaults to 6.
        grid (str, optional): 
            The centering of the phase mask array. Odd means cenetered on a single pixel, 
            even means cenetered on the quadrant between four pixels. Defaults to 'odd'.

    Returns:
        ndarray: 
            Complex array representing the phasor of the vortex phase mask. 
    """
    
    if grid=='odd':
        x = xp.linspace(-npix//2, npix//2-1, npix)
    elif grid=='even':
        x = xp.linspace(-npix//2, npix//2-1, npix) + 1/2
    x,y = xp.meshgrid(x,x)
    th = xp.arctan2(y,x)

    phasor = xp.exp(1j*charge*th)
    
    return phasor

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

def get_dz_fresnel_tf(dz, N, wavelength, fnum):
    """
    Get the Fresnel transfer function for a shift dz from focus. This transfer function
    is a form of defocus that will be applied to the pupil plane wavefront to compute a 
    defocused focal plane wavefront. 

    Args:
        dz (float): 
            The shift from nominal focus in units of meters. 
        N (int): 
            The number of pixels across the array to compute the transfer function for. 
        wavelength (float):
            The wavelength to compute the transfer function for in units of meters. 
        fnum (float): 
            The F-number of the beam to the focal plane. 

    Returns:
        ndarray: 
            The transfer function as a complex phasor. 
    """

    df = 1.0 / (N * wavelength * fnum)
    rp = get_scaled_coords(N,df, shift=False)[-1]
    return xp.exp(-1j*np.pi*dz*wavelength*(rp**2))





