
import numpy as np
import scipy
import astropy.units as u
from datetime import datetime
today = int(datetime.today().strftime('%Y%m%d'))
import importlib
from importlib import reload
from astropy.io import fits
import h5py
import copy
import time
import tomlkit
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, CenteredNorm
from matplotlib.patches import Circle, Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

import cupy as cp

import esc_llowfsc_sim
from esc_llowfsc_sim.math_module import xp, xcipy, ensure_np_array
from esc_llowfsc_sim import utils, detector, dm, wfe, source_flux
import esc_llowfsc_sim.esc_fraunhofer as esc
from esc_llowfsc_sim import llowfsc_sim as llowfsc

from magpyx.utils import ImageStream
from esc_llowfsc_sim import shmim_utils

import argparse

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run a script with specific variables.")

    # Add arguments (optional arguments use a double dash prefix)
    parser.add_argument('--gpu_ind', type=int, default=7)
    parser.add_argument('--Navg_dm', type=int, default=100)
    parser.add_argument('--Noffload', type=int, default=0)
    parser.add_argument('--tt_gain', type=float, default=0.9)
    parser.add_argument('--zer_gain', type=float, default=0.3)
    parser.add_argument('--target_star_mag', type=float, default=5.0)
    parser.add_argument('--sim_duration', type=float, default=10)
    parser.add_argument('--save_data', type=int, default=0)
    # parser.add_argument('', type=, default=)

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments using dot notation (e.g., args.var1)
    gpu_ind = args.gpu_ind
    Navg_dm = args.Navg_dm
    Noffload = args.Noffload
    tt_gain = args.tt_gain
    zer_gain = args.zer_gain
    target_star_mag = args.target_star_mag
    tmax = args.sim_duration
    save_data = args.save_data

    print(f'Noffload : {Noffload}')
    print(f'tt_gain : {tt_gain}')
    print(f'zer_gain : {zer_gain}')
    print(f'target_star_mag : {target_star_mag}')
    # print(f' : {}')

cp.cuda.Device(gpu_ind).use()

model_path = esc_llowfsc_sim.path/'data/2k-256/'

with open(model_path/'model_params.toml', mode="r",) as fp:
    all_params = tomlkit.load(fp).unwrap()
model_params = all_params['model']
wfe_params = all_params['wfe']
camlo_params = all_params['camlo']
camsci_params = all_params['camsci']

Nwaves = 101
bw = 0.20
waves = np.linspace(model_params['wavelength_c']*(1 - bw/2), model_params['wavelength_c']*(1 + bw/2), Nwaves)

mag0_source_params = source_flux.mag0_source_params
mag0_source_params.update({'wavelengths':waves})
mag0_source = source_flux.SOURCE(**mag0_source_params)
mag0_source.plot_spectrum_ph()
mag0_flux_per_wave = mag0_source.calc_fluxes()
mag0_total_ep_flux = np.sum(mag0_flux_per_wave)
print(f'Total flux over bandpass = {mag0_total_ep_flux:.3e}')

M = esc.single(
    **model_params,
    entrance_flux=mag0_total_ep_flux,
)
M.load_wfe(model_path)
PREFPM_OPD0 = copy.copy(M.PREFPM_OPD)

M.CAMLO_CAMSCI_FLUX_RATIO = 5

camsci_module = importlib.import_module(camsci_params['module'])
camsci_cls = getattr(camsci_module, camsci_params['type'])
CAMSCI = camsci_cls(**camsci_params['params'])
M.CAMSCI = CAMSCI

camlo_module = importlib.import_module(camlo_params['module'])
camlo_cls = getattr(camlo_module, camlo_params['type'])
CAMLO = camlo_cls(**camlo_params['params'])
M.CAMLO = CAMLO

M.use_vortex = 1
camsci_im, camlo_im = M.snap_camsci_and_camlo()



