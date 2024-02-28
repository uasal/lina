from .math_module import xp, _scipy, cupy_avail
if cupy_avail:
    import cupy as cp
else:
    cp = False

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

import hcipy as hp
import poppy as pp
# import matlab.engine as mat
from scipy.ndimage import gaussian_filter
from skimage.transform import resize, downscale_local_mean
from skimage.filters import threshold_otsu
from copy import deepcopy

from functools import partial
import multiprocessing as mp
from time import sleep

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fdpr2')

import numpy as np
try:
    import cupy as cp
except ImportError:
    logger.warning('Could not import cupy. You may lose functionality.')
    cp = None

from scipy.optimize import minimize
from scipy.ndimage import binary_erosion #, shift, center_of_mass
from poppy import zernike


def fdpr(fit_mask, images, defocus_values, tol=1e-6, reg=0, wreg=10):

    # PV to RMS and sign convention change
    Ediv = get_defocus_probes(fit_mask, cp.asarray(defocus_values))

    # phase retrieval parameters
    modes = cp.asnumpy(pp.zernike.arbitrary_basis(fit_mask, nterms=37, outside=0))

    # square-ify the PSFs
    dims = int(np.sqrt(images[0].shape))
    psfs_sq = cp.asarray([images[0].to_dict()['values'].reshape(dims, dims), 
                          images[1].to_dict()['values'].reshape(dims, dims),
                          images[2].to_dict()['values'].reshape(dims, dims)])

    # run phase retrieval
    prdict = run_phase_retrieval(psfs_sq, fit_mask, tol, reg, wreg, Ediv, modes=modes, fit_amp=False)

    return prdict





def get_array_module(arr):
    if cp is not None:
        return cp.get_array_module(arr)
    else:
        return np

def forward_model(pupil, Eprobes, Eab):
    xp = get_array_module(Eab)
    Epupils = pupil * Eab * Eprobes # (k,y,x)
    Epupils /= xp.mean(xp.abs(Epupils),axis=(-2,-1))[:,None,None]
    Efocals = fft2_shiftnorm(Epupils, axes=(-2,-1)) # k simultaneous FFTs
    Ifocals = xp.abs(Efocals)**2
    return Ifocals, Efocals, Epupils
    
def get_err(Imeas, Imodel, weights):
    xp = get_array_module(Imeas)
    K = len(weights)
    t1 = xp.sum(weights * Imodel * Imeas, axis=(-2,-1))**2
    t2 = xp.sum(weights * Imeas**2, axis=(-2,-1))
    t3 = xp.sum(weights * Imodel**2, axis=(-2,-1))
    return 1 - 1/K * np.sum(t1/(t2*t3), axis=0)

def get_Ibar_model(Imeas, Imodel, weights):
    xp = get_array_module(Imeas)
    K = len(weights)
    t1 = xp.sum(Imeas*Imodel*weights, axis=(-2,-1))[:,None,None]
    t2 = xp.sum(weights*Imeas**2, axis=(-2,-1))[:,None,None]
    t3 = xp.sum(weights*Imodel**2, axis=(-2,-1))[:,None,None]
    return 2/K * weights * t1 / (t2 * t3**2) * (Imodel * t1 - Imeas * t3)

def get_grad(Imeas, Imodel, Efocals, Eprobes, Eab, A, phi, weights, pupil, fit_amp=True):
    xp = get_array_module(Imeas)
    
    # common gradient terms
    Ibar = get_Ibar_model(Imeas, Imodel, weights)
    Ehatbar = 2 * Efocals * Ibar
    Ebar = ifft2_shiftnorm(Ehatbar, axes=(-2,-1))
    
    # --- get Eab ---
    Eabbar = Ebar * Eprobes.conj()
    # get amplitude
    expiphi = np.exp(1j*phi)
    Abar = Eabbar * expiphi.conj()
    # get phase
    expiphibar = Eabbar * A
    phibar = xp.imag(expiphibar * expiphi.conj())

    # --- get E probe ---
    #(save for later for now)
    #Epbar = Ebar * Eab.conj()
    #phipbar = xp.imag(Epbar * Eprobes.conj())
    #abar = xp.sum(phipbar * phiprobes, axis=(-2,-1))
    
    # sum terms (better be purely real, should double check this!!!)
    gradA = xp.sum(Abar, axis=0).real
    gradphi = xp.sum(phibar, axis=0).real
    #grada = xp.sum(abar, axis=0).real
    
    '''if zmodes is not None: # project onto zernikes
        coeffsA = xp.sum(gradA*zmodes, axis=(-2,-1)) / zmodes[0].sum()
        coeffsphi = xp.sum(gradphi*zmodes, axis=(-2,-1)) / zmodes[0].sum()
        gradA = xp.sum(coeffsA[:,None,None]*zmodes,axis=0)
        gradphi = xp.sum(coeffsphi[:,None,None]*zmodes,axis=0)'''
    
    if fit_amp:
        return gradA, gradphi
    else:
        return gradphi

def get_sqerr_grad(params, pupil, mask, Eprobes, weights, Imeas, N, lambdap, modes, fit_amp):
    
    xp = get_array_module(Eprobes)
    
    # CPU to GPU if needed
    if xp is cp and isinstance(params, np.ndarray):
        params = cp.array(params)
    
    # params to wavefront
    #param_a = params[0]
    if fit_amp:
        params_amp = params[:N]
        params_phase = params[N:]
    else:
        params_amp = np.ones(N)
        params_phase = params[:N]
        
    #Eab = xp.zeros(mask.shape, dtype=complex)
    A = xp.zeros(mask.shape)
    phi = xp.zeros(mask.shape)
    if modes is None:
        A[mask] = params_amp
        phi[mask] = params_phase
    else:
        if fit_amp:
            A = xp.sum(modes * params_amp[:,None,None], axis=0)
        else:
            A = pupil.astype(float)
        phi = xp.sum(modes * params_phase[:,None,None], axis=0)
    
    # probe
    #Eprobes = np.exp(1j*param_a*phiprobes)
    
    Eab = A * np.exp(1j*phi)
    #Eab[mask] = params_re + 1j*params_im
    
    #print(A.max(), phi.max())
    
    # forward model
    Imodel, Efocals, Epupils = forward_model(pupil, Eprobes, Eab)
    #return Imodel
    
    # lsq error
    #err = np.sum(weights * np.sqrt( (Imodel - Imeas)**2 ))
    err = get_err(Imeas, Imodel, weights) + lambdap * xp.sum(params**2) 
    
    # update mindict
    '''smoothing = None
    if mindict['iter'] > mindict['niter_zmodes']:
        zmodes = None
        smoothing = mindict['smoothing']
        if smoothing is None:
            pass
        else:
            smoothing = smoothing / mindict['iter_smoothing']**mindict['smoothing_exp']
            if smoothing < mindict['smoothing_min']:
                smoothing = mindict['smoothing_min']
            mindict['iter_smoothing'] += 1
    mindict['iter'] += 1'''
    
    # gradient
    if fit_amp:
        gradA, gradphi = get_grad(Imeas, Imodel, Efocals, Eprobes, Eab, A, phi, weights, pupil, fit_amp=True)#[mask]

        if modes is None:
            grad_Aphi = xp.concatenate([#cp.asarray([grada,]),
                                    gradA[mask],gradphi[mask]], axis=0) + 2 * lambdap * params
        else:
            gradAmodal = xp.sum(gradA*modes,axis=(-2,-1))
            gradphimodal = xp.sum(gradphi*modes, axis=(-2,-1))
            grad_Aphi = xp.concatenate([gradAmodal, gradphimodal], axis=0) + 2 * lambdap * params
    else:
        gradphi = get_grad(Imeas, Imodel, Efocals, Eprobes, Eab, A, phi, weights, pupil, fit_amp=False)

        if modes is None:
            grad_Aphi = gradphi[mask] + 2 * lambdap * params
        else:
            gradphimodal = xp.sum(gradphi*modes, axis=(-2,-1))
            grad_Aphi = gradphimodal + 2 * lambdap * params
    
    # back to CPU
    if xp is cp:
        err = cp.asnumpy(err)
        grad_Aphi = cp.asnumpy(grad_Aphi)
    
    return err, grad_Aphi

def get_han2d_sq(N, fraction=1./np.sqrt(2), normalize=False):
    '''
    Radial Hanning window scaled to a fraction 
    of the array size.
    
    Fraction = 1. for circumscribed circle
    Fraction = 1/sqrt(2) for inscribed circle (default)
    '''
    #return np.sqrt(np.outer(np.hanning(shape[0]), np.hanning(shape[0])))

    # get radial distance from center

    # scale radial distances
    xp = cp

    x = xp.linspace(-N/2., N/2., num=N)
    rmax = N * fraction
    scaled = (1 - x / rmax) * xp.pi/2.
    window = xp.sin(scaled)**2
    window[xp.abs(x) > rmax] = 0
    return xp.outer(window, window)

def run_phase_retrieval(Imeas, fitmask, tol, reg, wreg, Eprobes, init_params=None, bounds=True, modes=None, fit_amp=True):

    xp = get_array_module(Imeas)

    # centroiding here? probably not.
    if modes is None:
        N = np.count_nonzero(fitmask)
    else:
        N = len(modes)

    # initialize pixel values if not given
    if init_params is None:
        if modes is None:
            fitsmooth = gauss_convolve(binary_erosion(fitmask, iterations=3), 3)
            init_params = np.concatenate([fitsmooth[fitmask],
                                          fitsmooth[fitmask]*0], axis=0)
        else:
            amp0 = np.zeros(len(modes))
            amp0[0] = 1
            ph0 = np.zeros(len(modes))
            init_params = np.concatenate([amp0, ph0], axis=0)
    # compute weights?
    weights = 1/(Imeas + wreg) * get_han2d_sq(Imeas[0].shape[0], fraction=0.7)
    weights /= np.max(weights,axis=(-2,-1))[:,None,None]

    if bounds:
        bounds = [(0,None),]*N + [(None,None),]*N
    else:
        bounds = None

    # get probes

    # force all to right kind of array (numpy or cupy)
    Eprobes = xp.asarray(Eprobes, dtype=xp.complex128)
    Imeas = xp.asarray(Imeas, dtype=xp.float64)
    weights = xp.asarray(weights, dtype=xp.float64)
    fitmask_cp = xp.asarray(fitmask)
    if modes is not None:
        modes_cp = xp.asarray(modes)
    else:
        modes_cp = None

    if not fit_amp:
        init_params = init_params[N:]
        bounds = bounds[N:]

    errfunc = get_sqerr_grad
    fitdict = minimize(errfunc, init_params, args=(fitmask_cp, fitmask_cp,
                        Eprobes, weights, Imeas, N, reg, modes_cp, fit_amp),
                        method='L-BFGS-B', jac=True, bounds=bounds,
                        tol=tol, options={'ftol' : tol, 'gtol' : tol, 'maxls' : 100})

    # construct amplitude and phase
    phase_est = np.zeros(fitmask.shape)
    amp_est = np.zeros(fitmask.shape)

    if fit_amp:
        if modes is None:
            phase_est[fitmask] = fitdict['x'][N:]
            amp_est[fitmask] = fitdict['x'][:N]
        else:
            phase_est = np.sum(fitdict['x'][N:,None,None] * modes, axis=0)
            amp_est = np.sum(fitdict['x'][:N,None,None] * modes, axis=0)
    else:
        if modes is None:
            phase_est[fitmask] = fitdict['x'][:N]
            amp_est = None
        else:
            phase_est = np.sum(fitdict['x'][:N,None,None] * modes, axis=0)
            amp_est = None

    return {
        'phase_est' : phase_est,
        'amp_est' : amp_est,
        'obj_val' : fitdict['fun'],
        'fit_params' : fitdict['x']
    }

# ------- multiprocessing -------

def _process_phase_retrieval_mpfriendly(fitmask, tol, reg, wreg, Eprobes, init_params, bounds, modes, Imeas):
    return run_phase_retrieval(Imeas, fitmask, tol, reg, wreg, Eprobes, init_params=init_params, bounds=bounds, modes=modes)

def multiprocess_phase_retrieval(allpsfs, fitmask, tol, reg, wreg, Eprobes, init_params=None, modes=None, bounds=True, processes=2, gpus=None):
    

    from functools import partial
    import multiprocessing as mp

    try:
        mp.set_start_method('spawn')
    except RuntimeError as e:
        logger.warning(e)

    # TO DO: figure out if this still needs to be sequential or can be the original multiprocess_phase_retrieval
    mpfunc = partial(_process_phase_retrieval_mpfriendly, fitmask, tol, reg, wreg, Eprobes, init_params, bounds, modes)

    # There's a weird possibly memory-related issue that prevents us from simply
    # splitting the full allpsfs hypercube across the processes for multiprocessing
    # using a Pool object (small inputs work, large inputs result in GPU blocking).
    # So I've implemented my own queue and worker system here.

    # available GPUs for processing
    if gpus is None:
        gpus = [0,]

    # with a single GPU, parsing the config file will yield an int instead
    if isinstance(gpus, int):
        gpus = [gpus,]

    # assign multiple processes per GPU
    gpu_list = gpus * processes
    ntot = len(allpsfs)

    # make the queue and pass all jobs in
    mpqueue = GPUQueue(gpu_list, mpfunc)
    for (i, psfcube) in enumerate(allpsfs):
        #print(psfcube.shape)
        #print(i)
        mpqueue.add([i, psfcube])

    # check for completion every second
    while len(mpqueue.raw_results) < ntot:
        sleep(1)

    # get the results back in order
    allresults = mpqueue.get_sorted_results()
    mpqueue.terminate()

    #return allresults
    return allresults

class GPUWorker(mp.Process):
    def __init__(self, queue_in, queue_out, gpu_id, func):
        mp.Process.__init__(self, args=(queue_in,))
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.gpu_id = gpu_id
        self.func = func
        self.logger = mp.get_logger()
        
    def run(self):
        with cp.cuda.device.Device(self.gpu_id):
            while True:
                task_id, task = self.queue_in.get()
                
                logger.info(f'Worker {self} starting task {task_id}.')
                result = self.func(task)
        
                #self.logger.info(f'{self.gpu_id} got a task.')
                self.queue_out.put([task_id, result])
            
class GPUQueue(object):
    def __init__(self, gpu_list, func):
        
        self._queue_in = mp.Queue()
        self._queue_out = mp.Queue()
        self._results = []
        
        # make and start a worker for each entry in gpu_list
        self.workers = []
        for gpu in gpu_list:
            self.workers.append(GPUWorker(self._queue_in, self._queue_out, gpu, func))
        for w in self.workers:
            w.start()

    def add(self, task):
        self._queue_in.put(task)
            
    @property
    def raw_results(self):
        '''
        Grab the unsorted results from the worker queue
        '''
        while not self._queue_out.empty():
            self._results.append(self._queue_out.get_nowait())
        return self._results
    
    def get_sorted_results(self):
        results = self.raw_results
        sort_idx = np.argsort(np.asarray([r[0] for r in results]))
        return np.asarray([r[1] for r in results])[sort_idx]
            
    def terminate(self, timeout=5):
        for w in self.workers:
            w.terminate()
            w.join(timeout=timeout)
            w.close()


# ------ SETUP --------

def get_defocus_probes(fitmask, vals_waves):
    zmodes = zernike.arbitrary_basis(fitmask, nterms=4, outside=0) 
    return cp.exp(1j*zmodes[-1]*2*cp.pi*cp.asarray(vals_waves)[:,None,None])

def get_fitting_region(shape, nside):
    cen = ( (shape[0]-1)/2., (shape[1]-1)/2.)
    yxslice = (slice( int(np.rint(cen[0]-nside/2.)), int(np.rint(cen[0]+nside/2.))),
               slice( int(np.rint(cen[1]-nside/2.)), int(np.rint(cen[1]+nside/2.))))
    mask = np.zeros(shape, dtype=bool)
    mask[yxslice] = 1
    return mask, yxslice #??

def get_amplitude_mask(amplitude, threshold_factor):
    thresh = threshold_otsu(amplitude)
    mask = amplitude > (thresh*threshold_factor)
    return mask


# ------- CONVENIENCE FUNCTIONS ---------

def get_gauss(sigma, shape, cenyx=None, xp=np):
    if cenyx is None:
        cenyx = xp.asarray([(shape[0])/2., (shape[1])/2.]) # no -1
    yy, xx = xp.indices(shape).astype(float) - cenyx[:,None,None]
    g = xp.exp(-0.5*(yy**2+xx**2)/sigma**2)
    return g / xp.sum(g)

def convolve_fft(in1, in2, force_real=False):
    out = ifft2_shiftnorm(fft2_shiftnorm(in1,norm=None)*fft2_shiftnorm(in2,norm=None),norm=None)
    if force_real:
        return out.real
    else:
        return out
    
def gauss_convolve(image, sigma, force_real=True):
    if cp is not None:
        xp = cp.get_array_module(image)
    else:
        xp = np
    g = get_gauss(sigma, image.shape, xp=xp)
    return convolve_fft(image, g, force_real=force_real)


def fft2_shiftnorm(image, axes=None, norm='ortho', shift=True):

    if axes is None:
        axes = (-2, -1)

    if isinstance(image, np.ndarray): # CPU or GPU
        xp = np
    else:
        xp = cp

    if shift:
        shiftfunc = xp.fft.fftshift
        ishiftfunc = xp.fft.ifftshift
    else:
        shiftfunc = ishiftfunc = lambda x, axes=None: x

    if isinstance(image, np.ndarray):
        t = np.fft.fft2(ishiftfunc(image, axes=axes), axes=axes, norm=norm) #pyfftw.builders.fft
        return shiftfunc(t,axes=axes)
    else:
        return shiftfunc(cp.fft.fft2(ishiftfunc(image, axes=axes), axes=axes, norm=norm), axes=axes)
    
def ifft2_shiftnorm(image, axes=None, norm='ortho', shift=True):

    if axes is None:
        axes = (-2, -1)

    if isinstance(image, np.ndarray): # CPU or GPU
        xp = np
    else:
        xp = cp

    if shift:
        shiftfunc = xp.fft.fftshift
        ishiftfunc = xp.fft.ifftshift
    else:
        shiftfunc = ishiftfunc = lambda x, axes=None: x

    if isinstance(image, np.ndarray):
        t =np.fft.ifft2(ishiftfunc(image, axes=axes), axes=axes, norm=norm) # pyfftw.builders.ifft2
        return shiftfunc(t, axes=axes)
    else:
        return shiftfunc(cp.fft.ifft2(ishiftfunc(image, axes=axes), axes=axes, norm=norm), axes=axes)
