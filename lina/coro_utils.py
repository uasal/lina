from .math_module import xp, xcipy, ensure_np_array
from esc_llowfsc_sim import utils

import numpy as np
import scipy
import time
import copy
import os
from pathlib import Path
from IPython.display import clear_output

try:
    from scoobpy import utils as scoob_utils
    import purepyindi
    import purepyindi2
    from magpyx.utils import ImageStream
    import ImageStreamIOWrap as shmio
except ImportError:
    print('SCoOB interface does not have the required packages to operate.')

def normalize_coro_im(raw_im, im_params, ref_params, dark_im=0.0):
    exp_time_factor = ref_params['exp_time'] / im_params['exp_time'] if 'exp_time' in ref_params.keys() else 1.0
    gain_factor = 10**(ref_params['gain']/20 * 0.1) / 10**(im_params['gain']/20 * 0.1) if 'gain' in ref_params.keys() else 1.0
    fiber_atten_factor = 10**(-ref_params['atten']/10) / 10**(-im_params['atten']/10) if 'atten' in ref_params.keys() else 1.0
    ds_im = (raw_im - dark_im) 
    ni_im = ds_im * exp_time_factor * gain_factor * fiber_atten_factor / ref_params['Imax']
    return ni_im

def compute_contrast(ni_im, mask, ):
    ni_im_masked = ni_im[mask] # select pixels in desired mask
    ni_im_gtz = ni_im_masked[ni_im_masked>0] # select values greater than zero
    contrast = np.mean(ni_im_gtz)
    return contrast

def move_psf(x_pos, y_pos, client):
    client.wait_for_properties(['stagepiezo.stagepupil_x_pos', 'stagepiezo.stagepupil_y_pos'])
    scoob_utils.move_relative(client, 'stagepiezo.stagepupil_x_pos', x_pos)
    time.sleep(0.25)
    scoob_utils.move_relative(client, 'stagepiezo.stagepupil_y_pos', y_pos)
    time.sleep(0.25)

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








