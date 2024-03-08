from lina.math_module import xp, _scipy, ensure_np_array
from lina import imshows

import numpy as np
import astropy.units as u
import time 

import poppy 
'''
POPPY is being used here as the physical optics propagator for the model, however,
the model can be written with whatever software the user wishes. It will simply need to be 
wrapped in a class with the required methods and attributes for the algorithms in lina.

To connect with a testbed, the interface for the testbed will also need to be wrapped in 
a similar class, although direct computation of electric fields will obviously not be 
possible in that case. 
'''

class CORONAGRAPH():

    def __init__(self, 
                 wavelength=None, 
                 npix=256, 
                 oversample=8,
                 npsf=128,
                 psf_pixelscale_lamD=1/4, 
                 dm_ref=np.zeros((34,34)),
                 Imax_ref=1.0,
                 WFE=None, APODIZER=None, FPM=None, LYOT=None,
                 ):
        """ This is an example coronagraph model to clarify all the required attributes 
        and methods for a model or interface to be compatible with all Lina algorithms

        Parameters
        ----------
        wavelength : astropy.quantity, optional
            REQUIRED, by default None
        npix : int, optional
            REQUIRED, number of pixels across the pupil, by default 256
        oversample : int, optional
            REQUIRED, the factor by which the pupil array is padded for improved focal plane sampling, by default 16
        npsf : int, optional
            REQUIRED, number of pixels across the image array, by default 100
        psf_pixelscale_lamD : _type_, optional
            REQUIRED, sampling of the image plane in terms of lambda/D, by default None
        dm_ref : np.ndarray, optional
            REQUIRED, the reference command for the DM to be reset to, by default np.zeros((34,34))
        Imax_ref : float, optional
            REQUIRED, the maximum irradiance to normalize the images to, by default 1
        """
        self.wavelength_c = 650e-9*u.m # REQUIRED
        self.pupil_diam = 10.0*u.mm # REQUIRED
        
        self.wavelength = self.wavelength_c if wavelength is None else wavelength # REQUIRED
        
        self.npix = npix
        self.oversample = oversample
        self.N = int(self.npix*self.oversample)

        self.npsf = npsf # REQUIRED
        self.psf_pixelscale_lamD = psf_pixelscale_lamD # REQUIRED
        
        self.Imax_ref = Imax_ref # REQUIRED
        
        self.dm_ref = dm_ref # REQUIRED
        
        self.Nact = 34 # REQUIRED
        self.Nacts = 952 # REQUIRED
        self.act_spacing = 300e-6*u.m # REQUIRED
        self.dm_active_diam = 10.2*u.mm # REQUIRED
        self.dm_full_diam = 11.1*u.mm
        self.full_stroke = 1.5e-6*u.m
        
        self.dm_mask = np.ones((self.Nact,self.Nact), dtype=bool) # REQUIRED (must make sure the DM mask for active actuators is defined)
        xx = (np.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2)
        x,y = np.meshgrid(xx,xx)
        r = np.sqrt(x**2 + y**2)
        self.dm_mask[r>self.Nact//2] = 0 # had to set the threshold to 10.5 instead of 10.2 to include edge actuators
        
        self.DM = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM', 
                                                   actuator_spacing=self.act_spacing, 
                                                   influence_func='inf.fits',
                                                   include_factor_of_two=True, 
                                                  ) # REQUIRED (can be any DM model or interface to DM hardware)

        # additional parameters specific to this optical model
        self.WFE = WFE
        self.APODIZER = APODIZER
        self.FPM = FPM
        self.LYOT = LYOT
        
        self.init_osys()
        
    def getattr(self, attr): 
        # REQUIRED if you wish to parallelize the model for broadband or polarization modeling
        return getattr(self, attr)
    
    def setattr(self, attr):
        # REQUIRED if you wish to parallelize the model for broadband or polarization modeling
        pass
        
    def reset_dm(self): # SUGGESTED
        self.set_dm(self.dm_ref)
    
    def zero_dm(self): # SUGGESTED
        self.set_dm(np.zeros((self.Nact,self.Nact)))
        
    def set_dm(self, dm_command):
        # REQUIRED
        self.DM.set_surface(ensure_np_array(dm_command))
        
    def add_dm(self, dm_command):
        # REQUIRED
        self.DM.set_surface(self.get_dm() + ensure_np_array(dm_command))
        
    def get_dm(self):
        # REQUIRED
        return ensure_np_array(self.DM.surface)
    
    def map_actuators_to_command(self, act_vector):
        # SUGGESTED
        command = np.zeros((self.Nact, self.Nact))
        command.ravel()[self.dm_mask.ravel()] = ensure_np_array(act_vector)
        return command
    
    def init_osys(self):
        '''
        This method is specific to this optical model.
        Implement the model or testbed interface however it is optimal within a class.
        '''
        WFE = poppy.ScalarTransmission(name='WFE Place-holder') if self.WFE is None else self.WFE
        APODIZER = poppy.ScalarTransmission(name='Apodizer Place-holder') if self.APODIZER is None else self.APODIZER
        FPM = poppy.ScalarTransmission(name='FPM Place-holder') if self.FPM is None else self.FPM
        LYOT = poppy.ScalarTransmission(name='Lyot Stop Place-holder') if self.LYOT is None else self.LYOT
        
        # define OpticalSystem and add optics
        osys = poppy.OpticalSystem(pupil_diameter=self.pupil_diam, npix=self.npix, oversample=self.oversample)
        
        osys.add_pupil(poppy.CircularAperture(radius=self.pupil_diam/2))
        osys.add_pupil(WFE)
        osys.add_pupil(self.DM)
        osys.add_image(poppy.ScalarTransmission('Intermediate Image Plane'))
        osys.add_pupil(APODIZER)
        osys.add_image(FPM)
        osys.add_pupil(LYOT)
        
        self.as_per_lamD = (self.wavelength.to_value(u.m)/self.pupil_diam.to_value(u.m) * u.radian).to(u.arcsec)
        self.psf_pixelscale_as = self.psf_pixelscale_lamD * self.as_per_lamD * self.oversample
        osys.add_detector(pixelscale=self.psf_pixelscale_as.value, fov_pixels=self.npsf/self.oversample)
        
        self.osys = osys
        
    def init_inwave(self):
        # Specific to this optical model
        inwave = poppy.Wavefront(diam=self.pupil_diam, wavelength=self.wavelength,
                                 npix=self.npix, oversample=self.oversample)
        self.inwave = inwave
    
    def calc_wfs(self, quiet=False):
        """ Calculate and return all the wavefront objects.
        This is specific to this optical model and useful for debugging. 

        Parameters
        ----------
        quiet : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_osys()
        self.init_inwave()
        _, wfs = self.osys.calc_psf(inwave=self.inwave, return_intermediates=True)
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
        wfs[-1].wavefront /= np.sqrt(self.im_norm)
        return wfs
    
    def calc_psf(self, quiet=True): # method for getting the PSF in photons
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_osys()
        self.init_inwave()
        _, wf = self.osys.calc_psf(inwave=self.inwave, return_final=True, return_intermediates=False)
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
        return wf[0].wavefront/np.sqrt(self.Imax_ref)
    
    def snap(self): 
        # REQUIRED
        self.init_osys()
        self.init_inwave()
        _, wf = self.osys.calc_psf(inwave=self.inwave, return_intermediates=False, return_final=True)
        image = wf[0].intensity
        image /= self.Imax_ref
        return image
    
class IdealAGPM(poppy.AnalyticOpticalElement):
    """ Defines an ideal vortex phase mask coronagraph.
    Parameters
    ----------
    name : string
        Descriptive name
    wavelength : float
        Wavelength in meters.
    charge : int
        Charge of the vortex
    """
    @poppy.utils.quantity_input(wavelength=u.meter)
    def __init__(self, name="unnamed AGPM ",
                 wavelength=1e-6 * u.meter,
                 charge=2,
                 singularity=None,
                 **kwargs):
        
        poppy.AnalyticOpticalElement.__init__(self, planetype=poppy.poppy_core.PlaneType.intermediate, **kwargs)
        self.name = name
        self.lp = charge
        self.singularity = singularity
        self.central_wavelength = wavelength
        
    def get_phasor(self, wave):
        """
        Compute the amplitude transmission appropriate for a vortex for
        some given pixel spacing corresponding to the supplied Wavefront
        """

#         if not isinstance(wave, Wavefront) and not isinstance(wave, FresnelWavefront):  # pragma: no cover
#             raise ValueError("AGPM get_phasor must be called with a Wavefront "
#                              "to define the spacing")
#         assert (wave.planetype != PlaneType.image)

        y, x = self.get_coordinates(wave)
        phase = xp.arctan2(y, x)

        AGPM_phasor = xp.exp(1.j * self.lp * phase) * self.get_transmission(wave)
        
        idx = xp.where(x==0)[0][0]
        idy = xp.where(y==0)[0][0]
        AGPM_phasor[idx, idy] = 0
        return AGPM_phasor

    def get_opd(self, wave):
        y, x = self.get_coordinates(wave)
        phase = xp.arctan2(y, x)
        return self.lp * phase * self.central_wavelength.to(u.meter).value / (2 * np.pi)

    def get_transmission(self, wave):
        y, x = self.get_coordinates(wave)
        
        if self.singularity is None:
            trans = xp.ones(y.shape)
        else:
            circ = poppy.InverseTransmission(poppy.CircularAperture(radius=self.singularity/2))
            trans = circ.get_transmission(wave)
        return trans
    
