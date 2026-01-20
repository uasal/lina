from .math_module import xp, xcipy, ensure_np_array
from lina import utils, shmim_utils

import numpy as np
import scipy
import time
import copy
import os
from pathlib import Path
from IPython.display import clear_output
import skimage

from matplotlib.patches import Circle
from matplotlib.colors import LogNorm

try:
    # from scoobpy import utils as scoob_utils
    import purepyindi
    import purepyindi2
except ImportError:
    print('SCoOB interface does not have the required packages to operate.')

def normalize_coro_im(raw_im, im_params, ref_params, dark_im=0.0):
    exp_time_factor = ref_params['exp_time'] / im_params['exp_time'] if 'exp_time' in ref_params.keys() else 1.0
    gain_factor = 10**(ref_params['gain']/20 * 0.1) / 10**(im_params['gain']/20 * 0.1) if 'gain' in ref_params.keys() else 1.0
    fiber_atten_factor = 10**(-ref_params['atten']/10) / 10**(-im_params['atten']/10) if 'atten' in ref_params.keys() else 1.0
    ds_im = (raw_im - dark_im) 
    ni_im = ds_im * exp_time_factor * gain_factor * fiber_atten_factor / ref_params['Imax']
    return ni_im

def compute_contrast(ni_im, mask, verbose=True):
    ni_im_masked = ni_im[mask] # select pixels in desired mask
    gtz_mask = ni_im_masked>0
    if verbose:
        Nmask = mask.sum()
        Npix_gtz = gtz_mask.sum()
        ratio = Npix_gtz / Nmask
        print(f'Ratio of pixels greater than zero versus total pixels: {Npix_gtz} / {Nmask} = {ratio:.2f}')
    ni_im_gtz = ni_im_masked[gtz_mask] # select values greater than zero
    contrast = np.mean(ni_im_gtz)
    return contrast

def snap_ni(STREAM, NFRAMES, im_params, ref_params, dark_im=0.0, fp_shift=None):
    raw_im = np.mean(STREAM.grab_many(NFRAMES), axis=0)
    ni_im = normalize_coro_im(raw_im, im_params, ref_params, dark_im)
    if fp_shift is not None:
        scipy.ndimage.shift(ni_im, (fp_shift[1], fp_shift[0]), order=0)
    return ni_im, raw_im

def move_psf(x_pos, y_pos, client, delay=0.5):
    client.wait_for_properties(['stagepiezo.stagepupil_x_pos', 'stagepiezo.stagepupil_y_pos'])
    # scoob_utils.move_relative(client, 'stagepiezo.stagepupil_x_pos', x_pos)
    client[f'stagepiezo.stagepupil_x_pos.target'] = client[f'stagepiezo.stagepupil_x_pos.current'] + x_pos
    time.sleep(delay)
    # scoob_utils.move_relative(client, 'stagepiezo.stagepupil_y_pos', y_pos)
    client[f'stagepiezo.stagepupil_y_pos.target'] = client[f'stagepiezo.stagepupil_y_pos.current'] + y_pos
    time.sleep(delay)

def home_block(client, delay=2):
    client.wait_for_properties(['stagelinear.home'])
    client['stagelinear.home.request'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def move_block_in(client, delay=2):
    client.wait_for_properties(['stagelinear.presetName'])
    client['stagelinear.presetName.block_in'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def move_block_out(client, delay=2):
    client.wait_for_properties(['stagelinear.presetName'])
    client['stagelinear.presetName.block_out'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def home_filter_stage(client, delay=2):
    client.wait_for_properties(['rotationStageCtrl.home'])
    client['rotationStageCtrl.home'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def set_filter_stage_velocity(vel, client, delay=2):
    client.wait_for_properties(['rotationStageCtrl.velocity'])
    client['rotationStageCtrl.velocity.target'] = vel
    time.sleep(delay)

def set_filter_stage_position(angle, client, delay=2):
    client.wait_for_properties(['rotationStageCtrl.absDeg'])
    client['rotationStageCtrl.absDeg.target'] = angle
    time.sleep(delay)

def switch_filter_stage(filter_index, client, delay=2):
    client.wait_for_properties(['rotationStageCtrl.stageGoto'])
    client[f'rotationStageCtrl.stageGoto.filter{filter_index}'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def set_zwo_bin(bin, client, delay=0.25):
    # update roi parameters
    client.wait_for_properties(['camsci.roi_region_bin_x' ,'camsci.roi_region_bin_y', 
                                'camsci.roi_set'])
    client['camsci.roi_region_bin_x.target'] = bin
    client['camsci.roi_region_bin_y.target'] = bin
    time.sleep(delay)
    client['camsci.roi_set.request'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def set_zwo_roi(xc, yc, npix, client, delay=0.25):
    # update roi parameters
    client.wait_for_properties(['camsci.roi_region_x', 'camsci.roi_region_y', 
                                'camsci.roi_region_h' ,'camsci.roi_region_w', 
                                # 'camsci.roi_region_bin_x' ,'camsci.roi_region_bin_y', 
                                'camsci.roi_set'])
    client['camsci.roi_region_x.target'] = xc
    client['camsci.roi_region_y.target'] = yc
    client['camsci.roi_region_h.target'] = npix
    client['camsci.roi_region_w.target'] = npix
    time.sleep(delay)
    client['camsci.roi_set.request'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def set_fiber_atten(value, client, delay=0.1):
    client['fiberatten.attenuation.target'] = value
    time.sleep(delay)
    print(f'Set the fiber attenuation to {value:.1f}')

def get_fiber_atten(client,):
    fib_atten = client['fiberatten.attenuation.target']
    return fib_atten

# CAMERA Functions
def set_cam_roi(xc, yc, npix, client, cam_name='camsci', bin_mode=2, delay=0.25):
    # update roi parameters
    client.wait_for_properties([
        f'{cam_name}.roi_region_x', f'{cam_name}.roi_region_y', 
        f'{cam_name}.roi_region_h' , f'{cam_name}.roi_region_w', 
        f'{cam_name}.roi_region_bin_x', f'{cam_name}.roi_region_bin_y', 
        f'{cam_name}.roi_set',
    ])
    client[f'{cam_name}.roi_region_bin_x.target'] = bin_mode
    client[f'{cam_name}.roi_region_bin_y.target'] = bin_mode
    client[f'{cam_name}.roi_region_x.target'] = xc
    client[f'{cam_name}.roi_region_y.target'] = yc
    client[f'{cam_name}.roi_region_h.target'] = npix
    client[f'{cam_name}.roi_region_w.target'] = npix
    time.sleep(delay)
    client[f'{cam_name}.roi_set.request'] = purepyindi.SwitchState.ON
    time.sleep(delay)
    print(f'Set {cam_name} ROI.')

def set_cam_exp_time(exp_time, client, cam_name='camsci', delay=0.25):
    client.wait_for_properties([f'{cam_name}.exptime'])
    client[f'{cam_name}.exptime.target'] = exp_time
    time.sleep(delay)
    print(f'Set the {cam_name} exposure time to {exp_time:.2e}s')

def get_cam_exp_time(client, cam_name):
    client.wait_for_properties([f'{cam_name}.exptime'])
    return client[f'{cam_name}.exptime.target']

def set_cam_gain(gain, client, cam_name='camsci', delay=0.1):
    client.wait_for_properties([f'{cam_name}.emgain'])
    client[f'{cam_name}.emgain.target'] = gain
    time.sleep(delay)
    print(f'Set the {cam_name} gain setting to {gain:.1f}')

def get_cam_gain(client, cam_name='camsci'):
    client.wait_for_properties([f'{cam_name}.emgain'])
    return client[f'{cam_name}.emgain.target']

def set_cam_blacklevel(val, client, cam_name='camsci', delay=0.1):
    client.wait_for_properties([f'{cam_name}.blacklevel'])
    client[f'{cam_name}.blacklevel.target'] = val
    time.sleep(delay)
    print(f'Set the {cam_name} blacklevel to {val:.1f}')

def get_im_params(client, cam_name, verbose=True):
    client.wait_for_properties([f'{cam_name}.exptime', f'{cam_name}.emgain', 'fiberatten.attenuation.target'])
    im_params = {
        'exp_time': get_cam_exp_time(client, cam_name), 
        'gain': get_cam_gain(client, cam_name),
        'atten': get_fiber_atten(client),
    }
    if verbose:
        print(
            f'Image parameters: \n\tExposure time = {im_params["exp_time"]:.2e} s \n\tGain = {im_params["gain"]:.0f} \n\tFiber attenuation = {im_params["atten"]:.1f}'
        )
    return im_params

def set_dm(STREAM, command, delay=0.05):
    STREAM.write(1e6*command)
    time.sleep(delay)

def add_dm(STREAM, command, delay=0.05):
    STREAM.write(STREAM.grab_latest() + 1e6*command)
    time.sleep(delay)

def zero_dm(STREAM, delay=0.05):
    STREAM.write(np.zeros(STREAM.shape))
    time.sleep(delay)

def measure_waffle_center_and_angle(
        waffle_im, 
        psf_pixelscale_lamD, 
        im_thresh=1e-4, 
        r_thresh_min=12,
        r_thresh_max=18, 
        verbose=True, 
        plot=True,
    ):
    npsf = waffle_im.shape[0]
    y,x = (xp.indices((npsf, npsf)) - npsf//2)*psf_pixelscale_lamD
    r = xp.sqrt(x**2 + y**2)
    waffle_mask = (waffle_im > im_thresh) * (r>r_thresh_min) * (r<r_thresh_max)

    centroids = []
    for i in [0,1]:
        for j in [0,1]:
            arr = waffle_im[j*npsf//2:(j+1)*npsf//2, i*npsf//2:(i+1)*npsf//2]
            mask = waffle_mask[j*npsf//2:(j+1)*npsf//2, i*npsf//2:(i+1)*npsf//2]
            cent = np.flip(skimage.measure.centroid(ensure_np_array(mask*arr)))
            cent[0] += i*npsf//2
            cent[1] += j*npsf//2
            centroids.append(cent)

    centroids.append(centroids[0])
    centroids = np.array(centroids)
    centroids[[2,3]] = centroids[[3,2]]
    if verbose: print('Centroids:\n', centroids)

    if plot: 
        patches = []
        for i in range(4): patches.append(Circle(centroids[i], 1, fill=False, color='black'))
        utils.imshow(
            [waffle_mask, waffle_im, waffle_mask*waffle_im], 
            norms=[LogNorm(np.max(waffle_im)/1e4)],
            all_patches=[patches],
        )

    mean_angle = 0.0
    for i in range(4):
        angle = np.arctan2(centroids[i+1][1] - centroids[i][1], centroids[i+1][0] - centroids[i][0]) * 180/np.pi
        if angle<0:
            angle += 360
        if 0<angle<90:
            angle = 90-angle
        elif 90<angle<180:
            angle = 180-angle
        elif 180<angle<270:
            angle = 270-angle
        elif 270<angle<360:
            angle = 360-angle
        mean_angle += angle/4
    if verbose: print('Angle: ', mean_angle)

    m1 = (centroids[0][1] - centroids[2][1])/(centroids[0][0] - centroids[2][0])
    m2 = (centroids[1][1] - centroids[3][1])/(centroids[1][0] - centroids[3][0])
    # print(m1,m2)
    b1 = -m1*centroids[0][0] + centroids[0][1]
    b2 =  -m2*centroids[1][0] + centroids[1][1]
    # print(b1,b2)

    # m1*x + b1 = m2*x + b2
    # (m1-m2) * x = b2 - b1
    xc = (b2 - b1) / (m1 - m2)
    yc = m1*xc + b1
    print('Measured center in X: ', xc)
    print('Measured center in Y: ', yc)

    xshift = np.round(npsf/2 - xc)
    yshift = np.round(npsf/2 - yc)
    print('Required shift in X: ', xshift)
    print('Required shift in Y: ', yshift)

    return xshift, yshift, mean_angle
