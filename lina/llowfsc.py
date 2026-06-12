from .math_module import xp, xcipy, ensure_np_array
from lina import rt_utils, utils

import numpy as np
import astropy.units as u
import copy
from IPython.display import display, clear_output
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

def acquire_ref(
        take_im_fun,
        take_im_params,
        wfs_mask,
        dark_im=0.0,
        flux_norm=True,
    ):
    """
    Acquire the reference image for LLOWFSC. 

    Args:
        take_im_fun (callable): 
            Function that returns the LLOWFSC camera image. 
        take_im_params (dict):
            Dictionary of additional parameters needed for the take_im_fun method
        wfs_mask (ndarray): 
            Binary mask defining the region of the interest on the camera.
        dark_im (float, optional): 
            Dark image to subtract from the reference image. Defaults to 0.0.
        flux_norm (bool, optional): 
            Normalize the image by the total counts within the wfs_mask ROI. Defaults to True.

    Returns:
        tuple: 
            Tuple containing the reference image as a 2D array 
            and the flux normalization coefficient

    """

    camlo_ref_im = take_im_fun(**take_im_params)

    camlo_ref_im_ds = camlo_ref_im - dark_im

    camlo_ref_im_ds *= wfs_mask

    if flux_norm:
        flux_norm_coeff = camlo_ref_im_ds[wfs_mask].sum()
        ref_im = camlo_ref_im_ds / flux_norm_coeff
    else:
        flux_norm_coeff = 1.0
        ref_im = camlo_ref_im_ds

    return ref_im, flux_norm_coeff

def calibrate_dm_modes(
        take_im_fun,
        take_im_params,
        set_dm_fun,
        set_dm_params,
        dm_modes, 
        wfs_mask, 
        amp=2e-9,
        base_command=None, 
        flux_norm_coeff=None,
        include_factor_2=False,
        plot=False,
        Nplot=None,
    ):

    """
    Method to calibrate LLOWFSC using central differences of a given modal basis. 
    The positive and negative of each mode is applied and the response is the difference
    image normalized by 2*amp. 

    Args:
        take_im_fun (callable): 
            Function that returns the LLOWFSC camera image. 
        take_im_params (dict):
            Dictionary of additional parameters needed for the take_im_fun method
        set_dm_fun (callable): 
            Function that applies the DM command to the coronagraph. First argument of 
            this function must be the DM command that will be applied. 
        set_dm_params (dict): 
            Dictionary of additional parameters needed for the set_dm_fun method.
        dm_modes (ndarray):
            Data cube containing the DM modes that will be calibrated. Shape must be 
            Nmodes X Nact X Nact.
        wfs_mask (ndarray):
            Binary mask defining the region of the interest on the camera.
        amp (float, optional):
            Amplitude to apply to each mode during calibration in meters. Defaults to 2e-9.
        base_command (ndarray, optional):
        flux_norm_coeff (float, optional):
        include_factor_of_2 (bool, optional):
        plot (bool, optional):
        Nplot (float, optional):

    Returns:
        tuple: 
            Tuple containing the response matrix along with the full response cube. The cube 
            contains each individual difference image before it has been vectorized using 
            the wfs_mask and is easier to visualize responses. 
    """
    
    Nmask = int(wfs_mask.sum())
    Nmodes = dm_modes.shape[0]
    Nact = dm_modes.shape[1]
    Ncamlo = wfs_mask.shape[0]

    if base_command is None: base_command = np.zeros((Nact, Nact))

    response_matrix = np.zeros((Nmask, Nmodes))
    response_cube = np.zeros((Nmodes, Ncamlo, Ncamlo))
    
    start = time.time()
    for i in range(Nmodes):
        dm_mode = amp*dm_modes[i]

        set_dm_fun(base_command + dm_mode, **set_dm_params)
        im_pos = take_im_fun(**take_im_params)

        set_dm_fun(base_command - dm_mode, **set_dm_params)
        im_neg = take_im_fun(**take_im_params)

        set_dm_fun(base_command, **set_dm_params)

        diff = im_pos - im_neg
        response_cube[i] = copy.copy(diff) / (2 * amp)
        response_matrix[:,i] = copy.copy(diff)[wfs_mask] / (2 * amp)
        
        if plot: 
            print(f"Calibrated mode {i+1:d}/{Nmodes:d} in {time.time()-start:.3f}s", end='')
            utils.imshow(
                [dm_mode, im_pos, diff], 
                titles=[f'Mode {i+1}', 'Positive Chop Image', 'Difference'], 
                cmaps=['viridis'],
                npix=[None, Nplot, Nplot]
            )
        else:
            print(f"\tCalibrated mode {i+1:d}/{Nmodes:d} in {time.time()-start:.3f}s", end='')
            print("\r", end="")

    if include_factor_2: # in case you want to account for reflection off your FSM/DM
        response_matrix /= 2

    if flux_norm_coeff is not None:
        response_matrix /= flux_norm_coeff

    return response_matrix, response_cube

def plot_responses(
        dm_modes, 
        response_cube, 
        figsize=(25,5),
        dpi=125,
        hspace=0.0,
        wspace=-0.05,
        title=None,
        title_fs=14,
    ):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(2, 10, figure=fig)
    fig.suptitle(title, fontsize=title_fs)

    for i in range(10):
        mode = ensure_np_array(dm_modes[i])
        response = ensure_np_array(response_cube[i])

        ax = fig.add_subplot(gs[0, i])
        ax.imshow(mode, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[1, i])
        ax.imshow(response, cmap='magma',)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(hspace=hspace, wspace=wspace)

def reconstruct(
        camlo_im, 
        ref_im, 
        wfs_mask,
        control_matrix,
        dark_im=0.0,
        modes=(0,10),
        flux_norm=True,
        return_del_im=False,
    ):

    """
    Compute the modal coefficients from a given LLOWFSC camera image. 

    Args:
        camlo_im (ndarray): 
            2D array of LLOWFSC camera image. 
        ref_im (ndarray):
            2D array of the reference image. 
        wfs_mask (ndarray):
            Binary mask defining the region of the interest on the camera.
        control_matrix (ndarray):
            Pseudo-inverted response matrix. Also known as reconstructor sometimes. 
        dark_im (float, optional): 
            Dark image to subtract from the CAMLO image. Defaults to 0.0.
        modes (tuple, optional):
            Tuple indicating which modal coefficients to reconstruct. Defaults to (0,10), 
            meaning the first 10 modal coefficients will be computed and returned. 
        flux_norm (bool, optional):
            Normalize the image by the total counts within the wfs_mask ROI. Defaults to True.
        return_del_im (bool, optional):
            Return the difference image along with the modal coefficients. 

    Returns:
        ndarray: 
            Vector of modal coefficients. 

            Note: If return_del_im is True, then a tuple will be returned containing 
            the modal coeffciients followed by the difference image. 
    """

    camlo_im_dark_sub = camlo_im - dark_im
    camlo_im_flux_norm = camlo_im_dark_sub / camlo_im_dark_sub[wfs_mask].sum() if flux_norm else camlo_im_dark_sub
    del_im = camlo_im_flux_norm - ref_im

    coeff = control_matrix[modes[0]:modes[1]].dot(del_im[wfs_mask])

    if return_del_im:
        return coeff, del_im*wfs_mask
    
    return coeff

def run(
        take_im_fun,
        take_im_params,
        set_dm_fun,
        set_dm_params,
        get_dm_fun,
        get_dm_params,
        get_gains,
        # get_gains_params,
        ref_im,
        control_matrix,
        dm_modes,
        wfs_mask,
        dark_im=0.0,
        get_zpo=None,
        get_zpo_params={},
        get_ffo=None,
        get_ffo_params={},
    ):

    """
    Run a single iteration of LLOWFSC. 

    Args:
        take_im_fun (callable): 
            Function that returns the LLOWFSC camera image. 
        take_im_params (dict):
            Dictionary of additional parameters needed for the take_im_fun method
        set_dm_fun (callable): 
            Function that applies the DM command to the coronagraph. First argument of 
            this function must be the DM command that will be applied. 
        set_dm_params (dict): 
            Dictionary of additional parameters needed for the set_dm_fun method.
        get_dm_fun (callable): 
            Function that reads in current DM command applied for LLOWFSC. 
        get_dm_params (dict): 
            Dictionary of additional parameters needed for the get_dm_fun method.
        get_gains (callable):
            Function to read in current gain values to use for LLOWFSC controller.
        ref_im (ndarray):
            2D array of the reference image. 
        control_matrix (ndarray):
            Pseudo-inverted response matrix. Also known as reconstructor sometimes. 
        dm_modes (ndarray):
            Data cube containing the DM modes that will be calibrated. Shape must be 
            Nmodes X Nact X Nact.
        wfs_mask (ndarray): 
            Binary mask defining the region of the interest on the camera.
        dark_im (float, optional): 
            Dark image to subtract from the reference image. Defaults to 0.0.
        get_zpo (callable, optional):
            Get the latest zero point offset which is applied as a correction to 
            the reference image so that desired DM commands are not 
            corrected by LLOWFSC. Defaults to None (no offset will be applied).
        get_zpo_params (dict, optional):
            Dictionary of parameters needed for get_zpo. 
        get_ffo (callable, optional):
            Get the latest feed forward offset which is applied as a correction to 
            the computed modal coefficients so that desired DM commands are not 
            corrected by LLOWFSC. Defaults to None (no offset will be applied).
        get_ffo_params (dict, optional):
            Dictionary of parameters needed for get_ffo.

    Note that get_zpo and get_ffo can be mathematically equivalent, but get_ffo is more 
    computationally efficient while get_zpo has the added benefit of visualizing the offset. 

    """

    camlo_im = take_im_fun(**take_im_params)

    zpo = get_zpo(**get_zpo_params) if get_zpo is not None else 0.0
    recon_coeff = reconstruct(
        camlo_im, 
        ref_im + zpo, 
        wfs_mask,
        control_matrix,
        dark_im=dark_im,
        modes=(0,10),
    )
    ffo = get_ffo(**get_ffo_params) if get_ffo is not None else 0.0
    recon_coeff -= ffo

    modal_coeff = - get_gains() * recon_coeff

    del_dm_command = np.sum(modal_coeff[:, None, None] * dm_modes, axis=0)

    total_dm_command = get_dm_fun(**get_dm_params) + del_dm_command

    set_dm_fun(total_dm_command, **set_dm_params)

def compute_zpo(
        DM_STREAMS,
        dm_mask,
        wfs_mask,
        response_matrix,
        dm_modal_matrix,
        ZPO_STREAM,
    ):

    """
    Compute the zero point offset for multiple DM commands specified 
    by providing the ImageStreams of the desired DM channels as a list. 
    The ZPO is returned as an array but also written to an ImageStream of its
    own to be visualized in real time. 

    Returns:
        DM_STREAMS (list): 
            List of ImageStream objects corresponding the DM channels we want
            offsets to be computed for. 
        dm_mask (ndarray): 
            binary mask specifying the active actuators of the DM array. 
        wfs_mask (ndarray):
            Binary mask defining the region of the interest on the camera.
        response_matrix (ndarray):
            2D array containing the response matrix from the LLOWFSC calibration. 
        dm_modal_matrix (ndarray):
            2D array containing the vectorized DM modes that the ZPO will
            be computed for. Must be of the shape Nmodes x Nacts. 
        ZPO_STREAM:
            ImageStream object. 
    """

    zpo = np.zeros((wfs_mask.shape[0], wfs_mask.shape[1]))
    for i in range(len(DM_STREAMS)):
        zpo[wfs_mask] += response_matrix.dot( dm_modal_matrix.dot(DM_STREAMS[i].grab_latest()[dm_mask]) )

    ZPO_STREAM.write(zpo)

    return zpo



