import numpy as np
import scipy

try:
    import cupy
    import cupyx.scipy
    cupy_avail = True
except ImportError:
    cupy_avail = False

try:
    import jax
    import jax.scipy
    jax_avail = True
except ImportError:
    jax_avail = False

class np_backend:
    """A shim that allows a backend to be swapped at runtime."""
    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == '_srcmodule':
            return self._srcmodule

        return getattr(self._srcmodule, key)
    
class scipy_backend:
    """A shim that allows a backend to be swapped at runtime."""
    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == '_srcmodule':
            return self._srcmodule

        return getattr(self._srcmodule, key)
    
try:
    import cupy as cp
    import cupyx.scipy
#     cp.cuda.Device(0).compute_capability
    cupy_avail = True
except ImportError:
    cupy_avail = False
    
xp = np_backend(cp) if cupy_avail else np_backend(np)
_scipy = scipy_backend(cupyx.scipy) if cupy_avail else scipy_backend(scipy)

def update_np(module):
    """_summary_

    Parameters
    ----------
    module : _type_
        _description_
    """
    xp._srcmodule = module
    
def update_scipy(module):
    """_summary_

    Parameters
    ----------
    module : _type_
        _description_
    """
    _scipy._srcmodule = module
        
def ensure_np_array(arr):
    if isinstance(arr, np.ndarray):
        return arr
    elif cupy_avail and isinstance(arr, cupy.ndarray):
        return arr.get()
    elif jax_avail and isinstance(arr, jax.numpy.ndarray):
        return np.asarray(arr)