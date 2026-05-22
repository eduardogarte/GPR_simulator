import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.signal import butter, filtfilt
from sklearn.cluster import DBSCAN
import time

from Demo_gpr import Tx_pulse, fase_rx

# Initial paramters

CONFIG = {
    #RF parameters
# 1 MHz   1000000
# 10 MHz  10000000
# 100 Mhz 100000000
# 1 Ghz   1000000000
    "freq_start": 100000000,
    "freq_end": 200000000,
    "T_sweep": 500000000,

    #Noise and filters
    "electric_noise": 5.0, # Thermal noise RX
    "noise_env": 1, # Cluter (wall)
    "tvg_alpha": 0.8,

    # DBSCAN
    "epsilon_dbscan": 0.6, # minimum distance between points [m]
    "min_samples": 20, # Minimum pixels to consider a target

    #Mine Geometry
    "z_pared": 0.5, # Wall distance
    "eps_stone": 6.0, # dieletric constant

    # Scenario. Position of targets
    "stones":[
        {"x": 2.5, "z": 1.5, "y": 1.8, "strenght": 1.0},
        {"x": 2.5, "z": 1.5, "y": 2.0, "strenght": 1.2},
        {"x": 2.5, "z": 1.5, "y": 2.0, "strenght": 0.8},
        {"x": 1.0, "z": 2.8, "y": 1.5, "strenght": 1.5}
    ]
}

##########

# class GPR_simulator():
#     def __init__(self):
#
#         #params
def Quantize_signal(signal):
    bits = 8
    nivels = 2** bits -1

    #Normalize
    center = np.mean(signal)
    subs = signal - center
    maxbin = np.max(subs) **2
    norm_sig = subs / maxbin

    q_signal = np.round(norm_sig / nivels) * maxbin

    r_signal = (q_signal / nivels) *maxbin +center

    return r_signal
def RF_signal(cfg):
    c = 3e8
    v_stone = c / np.sqrt(cfg["epsilon_stone"])
    B = cfg["freq_end"] - cfg["freq_start"]
    Fs = 1000000000
    Fs_rf = 2 * Fs
    t = np.arange(0, cfg["T_sweep"], 1 / Fs_rf)

    Tx_pulse = np.cos(2 * np.pi * (cfg["freq_start"] * t + (B / (2 * cfg["T_sweep"])) * t **2))
    b_filter, a_filter = butter(4, 25e6 / (0.5 * Fs_rf), btype="low")

    return t, Tx_pulse, b_filter, a_filter, B, v_stone, Fs_rf

def scanning_2d(cfg, t, Tx_pulse, b_filter, a_filter, B, v_stone, Fs_rf):
    x_antenna = np.linspace(0, 5, 150)
    N = len(t)
    b_scan_raw = np.zeros((N//2), len(x_antenna))

    for i, x_ant in enumerate(x_antenna):
        Rx_Total = np.zero_like(t)
        for stone in cfg["stones"]:
            dit = np.sqrt((x_ant- stone["x"])**2 +stone["z"] **2)
            tau = 2 * dit /v_stone
            t_rx = t-tau
            valid = t_rx>=0
            phase_rx = np.zeros_like(t)
            fase_rx[valid] = 2 * np.pi * (cfg["freq_start"] * t_rx[valid] + (B / (2 * cfg["T_sweep"])) * t_rx[valid]**2)
            Rx_Total[valid] += (stone["strenght"] / (dit**1.5)) * np.cos(phase_rx[valid])
            Rx_Total += np.random.uniform(0, cfg["noise_env"]) * Tx_pulse
            Rx_Total += np.random.normal(0, cfg["electric_noise"], N)

            Mix_signal = Tx_pulse * Rx_Total
            filt_signal = filtfilt(b_filter, a_filter, Mix_signal)
            q_signal = Quantize_signal(filt_signal)
            b_scan_raw[:, i] = np.abs(np.fft.fft(q_signal))[:N //2]

        return b_scan_raw, x_antenna


def pc_processing(cfg):
    nx, ny, nz = 50, 40 , 60
    x_grid = np.linspace(0, 5, nx)
    y_grid = np.linspace(0, 3, ny)
    z_grid = np.linspace(0, 4, nz)
    Z, Y, X = np.meshgrid(z_grid, y_grid, x_grid, indexing= 'ij')
    vol = np.zeros_like(Z)

    for stone in cfg["stones"]:
        dist = np.sqrt((X - stone["x"])**2 + (Y - stone["y"])**2 + (Z - stone["z"])**2)
        vol += 2.0 *np.sqrt(-(dist**2) / 0.05)

    #Noise stage
    vol += np.random.normal(0, cfg["electric_noise"], vol.shape)
    vol += np.random.uniform(0, cfg["noise_env"], vol.shape)
    vol = np.clip(vol, 0, None)
    vol_sr = ndimage.gaussian_filter(vol, sigma= 1.5)
    thd = np.max(vol_sr) *0.5
    z_p, y_p, x_p = np.where(vol_sr>thd)
    physic_pc = np.vstack([x_grid[x_p], y_grid[y_p], z_grid[z_p]]).T

    #DBSCAN
    dbscan = DBSCAN(eps = cfg["epsilon_dbscan"], min_samples=cfg["min_samples"])
    labels = dbscan.fit_predict(physic_pc)



    return physic_pc, labels, x_grid, y_grid, z_grid


def migration(b_scan_raw):
    frequencies = np.fft.fftfreq(N, 1/ Fs_rf)[:N //2]
    time_depth = frequencies * T_sweep / B
    depths = np.zeros_like(time_depth)
    tau_wall = 2 * z_pared / c

    for k, tau in enumerate(time_depth):
        if tau <= tau_wall:
            depths[k] = tau * c / 2
        else:
            depths[k] = z_pared + (tau - tau_wall) * v_roca / 2

    idx_max = np.argmax(depths > 4.0)
    b_scan_raw = b_scan_raw[:idx_max, :]
    depths = depths[:idx_max]
    # KIrchoff migration

    b_scan_migration = np.zeros_like(b_scan_raw)

    # Radargram
    for i, xout in enumerate(x_antenna):
        for j, zout in enumerate(depths):
            # Hiperbole equation
            dist_curva = np.sqrt(zout ** 2 + (x_antenna - xout) ** 2)
            # index search by the radargram
            index_z = np.searchsorted(depths, dist_curva)
            valid_index = index_z < len(depths)

            if np.any(valid_index):
                energy = np.sum(b_scan_raw[index_z[valid_index], np.arange(len(x_antenna))[valid_index]])
                b_scan_migration[j, i] = energy

    return b_scan_migration


def super_resolution(b_scan_migration):
    factor = 10
    b_scan_super_res = ndimage.zoom(b_scan_migration, zoom=(factor, factor), order=3)
    noise_th = np.max(b_scan_super_res) * 0.4
    b_scan_super_res = np.clip(b_scan_super_res - noise_th, 0, None)

    # #new axis
    x_axis_sr = np.linspace(0, 5, len(x_antenna) * factor)
    z_axis_sr = np.linspace(0, depths[-1], len(depths) * factor)

    ############################# Krigin for peaks detection
    th_atr = np.max(b_scan_super_res) * 0.7  ## remainign percertange of energy 30 %
    neighbours = 15
    max_val_image = ndimage.maximum_filter(b_scan_super_res, size=neighbours)

    peaks_mask = (b_scan_super_res == max_val_image) & (b_scan_super_res > th_atr)
    index_z_sr, index_x_sr = np.where(peaks_mask)
    # X_SR, Z_SR = np.meshgrid(grid_x_hd, grid_z_hd)

    X_SR, Z_SR = np.meshgrid(x_axis_sr, z_axis_sr)
    detected_targets = []
    for i in range(len(index_x_sr)):
        x_det = x_axis_sr[index_x_sr[i]]
        z_det = z_axis_sr[index_z_sr[i]]
        detected_targets.append((x_det, z_det))

    return detected_targets