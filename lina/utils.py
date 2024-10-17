from .math_module import xp, _scipy, ensure_np_array
from lina.imshows import imshow1, imshow2, imshow3
import threading as th

import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
import poppy
import pickle

import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle, Rectangle
from IPython.display import display, clear_output

class Process(th.Timer):  
    def run(self):  
        while not self.finished.wait(self.interval):  
            self.function(*self.args, **self.kwargs)
# process = Repeat(0.1, print, ['Repeating']) 
# process.start()
# time.sleep(5)
# process.cancel()

def make_grid(npix, pixelscale=1, half_shift=False):
    if half_shift:
        y,x = (xp.indices((npix, npix)) - npix//2 + 1/2)*pixelscale
    else:
        y,x = (xp.indices((npix, npix)) - npix//2)*pixelscale
    return x,y

def pad_or_crop( arr_in, npix ):
    n_arr_in = arr_in.shape[0]
    if n_arr_in == npix:
        return arr_in
    elif npix < n_arr_in:
        x1 = n_arr_in // 2 - npix // 2
        x2 = x1 + npix
        arr_out = arr_in[x1:x2,x1:x2].copy()
    else:
        arr_out = xp.zeros((npix,npix), dtype=arr_in.dtype)
        x1 = npix // 2 - n_arr_in // 2
        x2 = x1 + n_arr_in
        arr_out[x1:x2,x1:x2] = arr_in
    return arr_out

def save_fits(fpath, data, header=None, ow=True, quiet=False):
    data = ensure_np_array(data)
    if header is not None:
        keys = list(header.keys())
        hdr = fits.Header()
        for i in range(len(header)):
            hdr[keys[i]] = header[keys[i]]
    else: 
        hdr = None
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdu.writeto(str(fpath), overwrite=ow) 
    if not quiet: print('Saved data to: ', str(fpath))

def load_fits(fpath, header=False):
    data = xp.array(fits.getdata(fpath))
    if header:
        hdr = fits.getheader(fpath)
        return data, hdr
    else:
        return data

# functions for saving python objects
def save_pickle(fpath, data, quiet=False):
    out = open(str(fpath), 'wb')
    pickle.dump(data, out)
    out.close()
    if not quiet: print('Saved data to: ', str(fpath))

def load_pickle(fpath):
    infile = open(str(fpath),'rb')
    pkl_data = pickle.load(infile)
    infile.close()
    return pkl_data  

def rms(data, mask=None):
    if mask is None:
        return xp.sqrt(xp.mean(xp.square(data)))
    else:
        return xp.sqrt(xp.mean(xp.square(data[mask])))

def rotate_arr(arr, rotation, reshape=False, order=1):
    if arr.dtype == complex:
        arr_r = _scipy.ndimage.rotate(xp.real(arr), angle=rotation, reshape=reshape, order=order)
        arr_i = _scipy.ndimage.rotate(xp.imag(arr), angle=rotation, reshape=reshape, order=order)
        rotated_arr = arr_r + 1j*arr_i
    else:
        rotated_arr = _scipy.ndimage.rotate(arr, angle=rotation, reshape=reshape, order=order)
    return rotated_arr

def interp_arr(arr, pixelscale, new_pixelscale, order=1):
    Nold = arr.shape[0]
    old_xmax = pixelscale * Nold/2

    x,y = xp.ogrid[-old_xmax:old_xmax - pixelscale:Nold*1j,
                   -old_xmax:old_xmax - pixelscale:Nold*1j]

    Nnew = int(np.ceil(2*old_xmax/new_pixelscale)) - 1
    new_xmax = new_pixelscale * Nnew/2

    newx,newy = xp.mgrid[-new_xmax:new_xmax-new_pixelscale:Nnew*1j,
                            -new_xmax:new_xmax-new_pixelscale:Nnew*1j]
    
    x0 = x[0,0]
    y0 = y[0,0]
    dx = x[1,0] - x0
    dy = y[0,1] - y0

    ivals = (newx - x0)/dx
    jvals = (newy - y0)/dy

    coords = xp.array([ivals, jvals])

    interped_arr = _scipy.ndimage.map_coordinates(arr, coords, order=order)
    return interped_arr

def generate_wfe(diam, 
                 npix=256, oversample=1, 
                 wavelength=500*u.nm,
                 opd_index=2.5, amp_index=2, 
                 opd_seed=1234, amp_seed=12345,
                 opd_rms=10*u.nm, amp_rms=0.05,
                 remove_modes=3, # defaults to removing piston, tip, and tilt
                 ):
    wf = poppy.FresnelWavefront(beam_radius=diam/2, npix=npix, oversample=oversample, wavelength=wavelength)
    wfe_opd = poppy.StatisticalPSDWFE(index=opd_index, wfe=opd_rms, radius=diam/2, seed=opd_seed).get_opd(wf)
    wfe_amp = poppy.StatisticalPSDWFE(index=amp_index, wfe=amp_rms*u.nm, radius=diam/2, seed=amp_seed).get_opd(wf)
    
    wfe_amp = xp.asarray(wfe_amp)
    wfe_opd = xp.asarray(wfe_opd)

    mask = poppy.CircularAperture(radius=diam/2).get_transmission(wf)>0
    Zs = poppy.zernike.arbitrary_basis(mask, nterms=remove_modes, outside=0)
    
    Zc_amp = lstsq(Zs, wfe_amp)
    Zc_opd = lstsq(Zs, wfe_opd)
    for i in range(3):
        wfe_amp -= Zc_amp[i] * Zs[i]
        wfe_opd -= Zc_opd[i] * Zs[i]

    mask = poppy.CircularAperture(radius=diam/2).get_transmission(wf)>0
    wfe_rms = xp.sqrt(xp.mean(xp.square(wfe_opd[mask])))
    wfe_opd *= opd_rms.to_value(u.m)/wfe_rms

    wfe_amp = wfe_amp*1e9 + 1

    wfe_amp_rms = xp.sqrt(xp.mean(xp.square(wfe_amp[mask]-1)))
    wfe_amp *= amp_rms/wfe_amp_rms

    wfe = wfe_amp * xp.exp(1j*2*np.pi/wavelength.to_value(u.m) * wfe_opd)
    wfe *= poppy.CircularAperture(radius=diam/2).get_transmission(wf)

    return wfe, mask

def lstsq(modes, data):
    """Least-Squares fit of modes to data.

    Parameters
    ----------
    modes : iterable
        modes to fit; sequence of ndarray of shape (m, n)
    data : numpy.ndarray
        data to fit, of shape (m, n)
        place NaN values in data for points to ignore

    Returns
    -------
    numpy.ndarray
        fit coefficients

    """
    mask = xp.isfinite(data)
    data = data[mask]
    modes = xp.asarray(modes)
    modes = modes.reshape((modes.shape[0], -1))  # flatten second dim
    modes = modes[:, mask.ravel()].T  # transpose moves modes to columns, as needed for least squares fit
    c, *_ = xp.linalg.lstsq(modes, data, rcond=None)
    return c

def create_zernike_modes(pupil_mask, nmodes=15, remove_modes=0):
    if remove_modes>0:
        nmodes += remove_modes
    zernikes = poppy.zernike.arbitrary_basis(pupil_mask, nterms=nmodes, outside=0)[remove_modes:]

    return zernikes

def make_f(h=10, w=6, shift=(0,0), Nact=34):
    f_command = xp.zeros((Nact, Nact))

    top_row = Nact//2 + h//2 + shift[1]
    mid_row = Nact//2 + shift[1]
    row0 = Nact//2 - h//2 + shift[1]

    col0 = Nact//2 - w//2 + shift[0] + 1
    right_col = Nact//2 + w//2 + shift[0] + 1

    rows = xp.arange(row0, top_row)
    cols = xp.arange(col0, right_col)

    f_command[rows, col0] = 1
    f_command[top_row,cols] = 1
    f_command[mid_row,cols] = 1
    return f_command

def make_ring(rad=15, Nact=34):
    y,x = (xp.indices((Nact, Nact)) - Nact//2 + 1/2)
    r = xp.sqrt(x**2 + y**2)
    ring = (rad-1/2<r) * (r < rad+1/2)
    ring = ring.astype(float)
    return ring

def map_acts_to_dm(actuators, dm_mask):
    Nact = dm_mask.shape[0]
    command = xp.zeros((Nact, Nact))
    command.ravel()[dm_mask.ravel()] = actuators
    return command

# Create control matrix
def WeightedLeastSquares(A, weight_map, nprobes=2, rcond=1e-1):
    control_mask = weight_map > 0
    w = weight_map[control_mask]
    for i in range(nprobes-1):
        w = xp.concatenate((w, weight_map[control_mask]))
    W = xp.diag(w)
    print(W.shape, A.shape)
    cov = A.T.dot(W.dot(A))
    return xp.linalg.inv(cov + rcond * xp.diag(cov).max() * xp.eye(A.shape[1])).dot( A.T.dot(W) )

def TikhonovInverse(A, rcond=1e-15):
    U, s, Vt = xp.linalg.svd(A, full_matrices=False)
    s_inv = s/(s**2 + (rcond * s.max())**2)
    return (Vt.T * s_inv).dot(U.T)

def beta_reg(S, beta=-1):
    # S is the sensitivity matrix also known as the Jacobian
    sts = xp.matmul(S.T, S)
    rho = xp.diag(sts)
    alpha2 = rho.max()

    control_matrix = xp.matmul( xp.linalg.inv( sts + alpha2*10.0**(beta)*xp.eye(sts.shape[0]) ), S.T)
    return control_matrix

def create_circ_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w//2), int(h//2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
        
    Y, X = xp.ogrid[:h, :w]
    dist_from_center = xp.sqrt((X - center[0] + 1/2)**2 + (Y - center[1] + 1/2)**2)

    mask = dist_from_center <= radius
    return mask

# Creating focal plane masks
def create_annular_focal_plane_mask(npsf, psf_pixelscale, 
                                    irad, orad,  
                                    edge=None,
                                    shift=(0,0), 
                                    rotation=0,
                                    plot=False):
    x = (xp.linspace(-npsf/2, npsf/2-1, npsf) + 1/2)*psf_pixelscale
    x,y = xp.meshgrid(x,x)
    r = xp.hypot(x, y)
    mask = (r > irad) * (r < orad)
    if edge is not None: mask *= (x > edge)
    
    mask = _scipy.ndimage.rotate(mask, rotation, reshape=False, order=0)
    mask = _scipy.ndimage.shift(mask, (shift[1], shift[0]), order=0)
    
    if plot:
        imshow1(mask)
        
    return mask

def create_box_focal_plane_mask(npsf, psf_pixelscale_lamD, width, height, x0=0, y0=0,):
    x = (xp.linspace(-npsf/2, npsf/2-1, npsf) + 1/2)*psf_pixelscale_lamD
    x,y = xp.meshgrid(x,x)
    mask = ( abs(x - x0) < width/2 ) * ( abs(y - y0) < height/2 )
    return mask > 0

def masked_rms(image,mask=None):
    return np.sqrt(np.mean(image[mask]**2))

def create_random_probes(rms, alpha, dm_mask, fmin=1, fmax=17, nprobes=3, 
                         plot=False,
                         calc_responses=False):
    # randomized probes generated by PSD
    shape = dm_mask.shape
    ndm = shape[0]

    probes = []
    for n in range(nprobes):
        fx = np.fft.rfftfreq(ndm, d=1.0/ndm)
        fy = np.fft.fftfreq(ndm, d=1.0/ndm)
        fxx, fyy = np.meshgrid(fx, fy)
        fr = np.sqrt(fxx**2 + fyy**2)
        spectrum = ( fr**(alpha/2.0) ).astype(complex)
        spectrum[fr <= fmin] = 0
        spectrum[fr >= fmax] = 0
        cvals = np.random.standard_normal(spectrum.shape) + 1j * np.random.standard_normal(spectrum.shape)
        spectrum *= cvals
        probe = np.fft.irfft2(spectrum)
        probe *= dm_mask * rms / masked_rms(probe, dm_mask)
        probes.append(probe.real)
        
    probes = xp.asarray(probes)/rms
    
    if plot:
        for i in range(nprobes):
            response = xp.abs(xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift( pad_or_crop(probes[i], 4*ndm) ))))
            imshow2(probes[i], response, pxscl2=1/4)
                
    return probes

def create_hadamard_modes(dm_mask): 
    Nacts = dm_mask.sum().astype(int)
    Nact = dm_mask.shape[0]
    np2 = 2**int(xp.ceil(xp.log2(Nacts)))
    hmodes = xp.array(scipy.linalg.hadamard(np2))
    
    had_modes = []

    inds = xp.where(dm_mask.flatten().astype(int))
    for hmode in hmodes:
        hmode = hmode[:Nacts]
        mode = xp.zeros((dm_mask.shape[0]**2))
        mode[inds] = hmode
        had_modes.append(mode)
    had_modes = xp.array(had_modes).reshape(np2, Nact, Nact)
    
    return had_modes
    
def create_fourier_modes(dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, 
                         rotation=0, 
                         fourier_sampling=0.75,
                         which='both', 
                         return_fs=False,
                         plot=False,
                         ):
    Nact = dm_mask.shape[0]
    nfg = int(xp.round(npsf * psf_pixelscale_lamD/fourier_sampling))
    if nfg%2==1: nfg += 1
    yf, xf = (xp.indices((nfg, nfg)) - nfg//2 + 1/2) * fourier_sampling
    fourier_cm = create_annular_focal_plane_mask(nfg, fourier_sampling, iwa-fourier_sampling, owa+fourier_sampling, edge=iwa-fourier_sampling, rotation=rotation)
    ypp, xpp = (xp.indices((Nact, Nact)) - Nact//2 + 1/2)

    sampled_fs = xp.array([xf[fourier_cm], yf[fourier_cm]]).T
    if plot: imshow1(fourier_cm, pxscl=fourier_sampling, grid=True)
    
    fourier_modes = []
    for i in range(len(sampled_fs)):
        fx = sampled_fs[i,0]
        fy = sampled_fs[i,1]
        if which=='both' or which=='cos':
            fourier_modes.append( dm_mask * xp.cos(2 * np.pi * (fx*xpp + fy*ypp)/Nact) )
        if which=='both' or which=='sin':
            fourier_modes.append( dm_mask * xp.sin(2 * np.pi * (fx*xpp + fy*ypp)/Nact) )
    
    if return_fs:
        return xp.array(fourier_modes), sampled_fs
    else:
        return xp.array(fourier_modes)

def create_fourier_probes(dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, 
                          rotation=0, 
                          fourier_sampling=0.75, 
                          shifts=None, nprobes=2,
                          use_weighting=False, 
                          plot=False,
                          ): 
    Nact = dm_mask.shape[0]
    cos_modes, fs = create_fourier_modes(dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, rotation,
                                        fourier_sampling=fourier_sampling, 
                                        return_fs=True,
                                        which='cos',
                                        )
    sin_modes = create_fourier_modes(dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, rotation,
                                    fourier_sampling=fourier_sampling, 
                                    which='sin',
                                    )
    nfs = fs.shape[0]

    probes = xp.zeros((nprobes, Nact, Nact))
    if use_weighting:
        fmax = xp.max(np.sqrt(fs[:,0]**2 + fs[:,1]**2))
        sum_cos = 0
        sum_sin = 0
        for i in range(nfs):
            f = np.sqrt(fs[i][0]**2 + fs[i][1]**2)
            weight = f/fmax
            sum_cos += weight*cos_modes[i]
            sum_sin += weight*sin_modes[i]
        sum_cos = sum_cos
        sum_sin = sum_sin
    else:
        sum_cos = cos_modes.sum(axis=0)
        sum_sin = sin_modes.sum(axis=0)
    
    # nprobes=2 will give one probe that is purely the sum of cos and another that is the sum of sin
    cos_weights = np.linspace(1,0,nprobes)
    sin_weights = np.linspace(0,1,nprobes)
    
    shifts = [(0,0)]*nprobes if shifts is None else shifts

    for i in range(nprobes):
        probe = cos_weights[i]*sum_cos + sin_weights[i]*sum_sin
        probe = _scipy.ndimage.shift(probe, (shifts[i][1], shifts[i][0]))
        probes[i] = probe/xp.max(probe)

        if plot: 
            probe_response = xp.abs(xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(pad_or_crop(probes[i], 4*Nact)))))
            imshow2(probes[i], probe_response, cmap1='viridis', pxscl2=1/4)

    return probes

def create_poke_probes(Nact, poke_indices, plot=False):
    Nprobes = len(poke_indices)
    probe_modes = np.zeros((Nprobes, Nact, Nact))
    for i in range(Nprobes):
        probe_modes[i, poke_indices[i][1], poke_indices[i][0]] = 1
    if plot:
        fig,ax = plt.subplots(nrows=1, ncols=1, dpi=125, figsize=(5,5))
        ax.scatter(poke_indices[0], poke_indices[1])
        ax.grid()
        plt.close()
        display(fig)
        
    return probe_modes

def create_all_poke_modes(dm_mask):
    Nact = dm_mask.shape[0]
    Nacts = int(np.sum(dm_mask))
    poke_modes = xp.zeros((Nacts, Nact, Nact))
    count=0
    for i in range(Nact):
        for j in range(Nact):
            if dm_mask[i,j]:
                poke_modes[count, i,j] = 1
                count+=1
    return poke_modes

def create_sinc_probe(Nacts, amp, probe_radius, probe_phase=0, offset=(0,0), bad_axis='x'):
    print('Generating probe with amplitude={:.3e}, radius={:.1f}, phase={:.3f}, offset=({:.1f},{:.1f}), with discontinuity along '.format(amp, probe_radius, probe_phase, offset[0], offset[1]) + bad_axis + ' axis.')
    
    xacts = np.arange( -(Nacts-1)/2, (Nacts+1)/2 )/Nacts - np.round(offset[0])/Nacts
    yacts = np.arange( -(Nacts-1)/2, (Nacts+1)/2 )/Nacts - np.round(offset[1])/Nacts
    Xacts,Yacts = np.meshgrid(xacts,yacts)
    if bad_axis=='x': 
        fX = 2*probe_radius
        fY = probe_radius
        omegaY = probe_radius/2
        probe_commands = amp * np.sinc(fX*Xacts)*np.sinc(fY*Yacts) * np.cos(2*np.pi*omegaY*Yacts + probe_phase)
    elif bad_axis=='y': 
        fX = probe_radius
        fY = 2*probe_radius
        omegaX = probe_radius/2
        probe_commands = amp * np.sinc(fX*Xacts)*np.sinc(fY*Yacts) * np.cos(2*np.pi*omegaX*Xacts + probe_phase) 
    if probe_phase == 0:
        f = 2*probe_radius
        probe_commands = amp * np.sinc(f*Xacts)*np.sinc(f*Yacts)
    return probe_commands

def create_sinc_probes(Npairs, Nact, dm_mask, probe_amplitude, probe_radius=10, probe_offset=(0,0), plot=False):
    probe_phases = np.linspace(0, np.pi*(Npairs-1)/Npairs, Npairs)
    
    probes = []
    for i in range(Npairs):
        if i%2==0:
            axis = 'x'
        else:
            axis = 'y'
            
        probe = create_sinc_probe(Nact, probe_amplitude, probe_radius, probe_phases[i], offset=probe_offset, bad_axis=axis)
            
        probes.append(probe*dm_mask)
    probes = np.array(probes)
    if plot:
        for i,probe in enumerate(probes):
            probe_response = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift( pad_or_crop(probe, int(4*Nact))  ))))
            imshow2(probe, probe_response, pxscl2=1/4)
    
    return probes
    
def get_radial_dist(shape, scaleyx=(1.0, 1.0), cenyx=None):
    '''
    Compute the radial separation of each pixel
    from the center of a 2D array, and optionally 
    scale in x and y.
    '''
    indices = np.indices(shape)
    if cenyx is None:
        cenyx = ( (shape[0] - 1) / 2., (shape[1] - 1)  / 2.)
    radial = np.sqrt( (scaleyx[0]*(indices[0] - cenyx[0]))**2 + (scaleyx[1]*(indices[1] - cenyx[1]))**2 )
    return radial

def get_radial_contrast(im, mask, nbins=50, cenyx=None):
    im = ensure_np_array(im)
    mask = ensure_np_array(mask)
    radial = get_radial_dist(im.shape, cenyx=cenyx)
    bins = np.linspace(0, radial.max(), num=nbins, endpoint=True)
    digrad = np.digitize(radial, bins)
    profile = np.asarray([np.mean(im[ (digrad == i) & mask]) for i in np.unique(digrad)])
    return bins, profile
    
def plot_radial_contrast(im, mask, pixelscale, nbins=30, cenyx=None, xlims=None, ylims=None):
    bins, contrast = get_radial_contrast(im, mask, nbins=nbins, cenyx=cenyx)
    r = bins * pixelscale

    fig,ax = plt.subplots(nrows=1, ncols=1, dpi=125, figsize=(6,4))
    ax.semilogy(r,contrast)
    ax.set_xlabel('radial position [$\lambda/D$]')
    ax.set_ylabel('Contrast')
    ax.grid()
    if xlims is not None: ax.set_xlim(xlims[0], xlims[1])
    if ylims is not None: ax.set_ylim(ylims[0], ylims[1])
    plt.close()
    display(fig)
    

