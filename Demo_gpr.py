import math

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.signal import butter, filtfilt
import scipy.ndimage as ndimage


def simular_adc_8bits(senal_analogica):
    """
    Simulate an ADC 8 bits, it discretize in 256 leves


    Parameters
    sa : ndarray, shape(M)
        Analog signal


    Returns
    senal_cuantizada : ndarray
        Quantized signal
    """
    v_ref_fijo = None
    bits = 8
    niveles = 2 ** bits - 1
    senal_analogica = senal_analogica / niveles
    # 3. Cuantización a 8 bits
    senal_cuantizada = np.floor(filt_signal / (1.0 - (-1.0))/ 2**8)

    return senal_cuantizada
# ==========================================
# Stage 1. System parameters (FMCW - Low frequency)
# ==========================================

print("Radar system initialed")
c= 3e8
eps_roca = 6.0
v_roca = c / np.sqrt(eps_roca)
# 1 MHz   1000000
# 10 MHz  10000000
# 100 Mhz 100000000
# 1 Ghz   1000000000
freq_start = 100000000 # Mhz
freq_end =   800000000 # Mhz
B = freq_end - freq_start # bandwidth gHz
T_sweep = 5e-6 # Scanning in microseconds
Fs = 400000000
Fs_rf =  2*Fs # ADC sampling
t = np.arange(0, T_sweep, 1/Fs_rf)
N = len(t) # samples

#Tx
A = 1.0
Tx_pulse = A * np.cos(2* np.pi * (freq_start * t + ( B /( 2* T_sweep)) * t**2))
b_filter, a_filter = butter(4, 25e6 / (0.05* Fs_rf), btype='low')

# plt.figure(figsize=(10, 8))
# plt.plot(t, Tx_pulse)
# plt.title('Tx Pulse')
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.show()

# ==========================================
# Stage 2. Environment simulation
# ==========================================

x_antenna = np.linspace(0, 5, 150) # antenna scanning in 5 meters of depth

# Stage 2.a Random creation of objects to find
# axis x: view field, axis z: depth, axis y: height
stones = [
    {"x": 1.0, "z":0.8, "y": 2.0, "radio": 0.15, "fuerza": 1.0},
    {"x": 2.5, "z": 1.5, "y": 2.0, "radio": 0.15, "fuerza": 1.2},
    {"x": 3.2, "z": 0.9, "y": 2.0, "radio": 0.10, "fuerza": 0.8}
]


######## 3D DATA
nx, ny, nz = 50, 40, 60
x_grid = np.linspace(0, 5, nx)
z_grid = np.linspace(0, 4, nz)
y_grid = np.linspace(0, 3, ny)

Z, Y, X = np.meshgrid(z_grid, y_grid, x_grid, indexing = 'ij')
vol_ = np.zeros_like(Z)

b_scan_raw = np.zeros((N // 2, len(x_antenna))) # raw radargram matrix

# adding
### Constructor for 3d data
for stone in stones:
    dit_3d = np.sqrt((X - stone["x"]) ** 2 + (Y - stone["y"]) ** 2 + (Z - stone["z"]) ** 2)
    vol_ += 2.0 * np.exp(-(dit_3d ** 2) / 0.05)
z_pared = 0.5
n=0
dit_p = []
print('Facade scanning and FMCW raw generation')
for i, x_ant in enumerate(x_antenna):
    RX_total = np.zeros_like(t)
    for stone in stones:

        dit = np.sqrt((x_ant - stone["x"])**2 + stone["z"]**2)        # if z_pixel <= z_pared :

        #     tau = 2* dit/c
        # else:
        #     L_air = dit * (z_pared/ z_pixel)
        #     L_stone = dit - L_air
        #     tau =2 * (L_air/ c + L_stone /v_roca)
        tau = 2 * dit / v_roca
        t_rx = t - tau
        valid = t_rx>=0
        fase_rx = np.zeros_like(t)
        fase_rx[valid] = 2 * np.pi * (freq_start * t_rx[valid] + (B / (2 * T_sweep)) * t_rx[valid]**2)
        reflectividad = stone["fuerza"] / (dit**1.5)
        RX_total[valid] += reflectividad * np.cos(fase_rx[valid])
    #NOISE environmental
    ruido_rugosidad = np.random.uniform(0, 5)
    RX_total += ruido_rugosidad * Tx_pulse  # Eco instantáneo desordenado
    #electric noise
    RX_total += np.random.normal(0, 2, N)
    Mix_sig = Tx_pulse * RX_total
    filt_signal = filtfilt(b_filter, a_filter, Mix_sig)
    digital_signal = simular_adc_8bits(filt_signal)
    b_scan_raw[:, i] = np.abs(np.fft.fft(digital_signal))[:N // 2]

# Stage 3 Migration

frequencies = np.fft.fftfreq(N, 1 / Fs_rf)[:N //2]
time_depth = frequencies * T_sweep / B
depths = np.zeros_like(time_depth)
tau_wall = 2* z_pared / c

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

#Radargram
for i, xout in enumerate(x_antenna):
    for j,zout in enumerate(depths):
        #Hiperbole equation
        dist_curva = np.sqrt(zout**2 + (x_antenna - xout)**2)
        #index search by the radargram
        index_z = np.searchsorted(depths, dist_curva)
        valid_index = index_z < len(depths)


        if np.any(valid_index):
            energy =np.sum(b_scan_raw[index_z[valid_index], np.arange(len(x_antenna))[valid_index]])
            b_scan_migration[j, i] = energy


plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.title('Raw Radargram')
plt.xlabel(" Position above the mine [m]")
plt.ylabel("Depth [m]")
plt.imshow(b_scan_raw, extent = [0, 5, 4, 0], aspect ='auto', cmap ='viridis')
plt.show()

# plt.figure()
plt.subplot(132)
plt.title('Migrated Radargram. Focused energy')
plt.xlabel(" Position above the mine [m]")
plt.ylabel("Depth [m]")
plt.imshow(b_scan_migration, extent=[0, 5, 4, 0], aspect ='auto', cmap ='inferno')
plt.scatter([r['x'] for r in stones], [r['z'] for r in stones], color='cyan', marker='+', s=120, label='backscattered energy')
plt.legend(loc='lower left')
plt.colorbar()
plt.show()
# nx, nz = b_scan_migration.shape
# x_grid = np.linspace(0, 5, nx)
# z_grid = np.linspace(0, 4, nz)
# b_scan_processed = np.zeros((nz, nx))
# #### A mesh for filled the center phase detected
# X, Z, = np.meshgrid(x_grid, z_grid)
# map_subsurface = np.zeros_like(X)
#
# real_points= [(s["x"], s["z"]) for s in stones]
#
# for rx, rz in real_points:
#     idx_x = np.argmin(np.abs(x_grid - rx))
#     idx_z = np.argmin(np.abs(z_grid - rz))
#     # Creamos un blob de energía alrededor del centro
#     for i in range(nx):
#         for j in range(nz):
#             dist_p = np.sqrt((x_grid[i] - rx) ** 2 + (z_grid[j] - rz) ** 2)
#             b_scan_processed[j, i] += np.exp(-(dist_p ** 2) / 0.05)
#
# # Noise
# b_scan_processed += np.random.normal(0, 0.2, (nz, nx))
# # Rectificamos para no tener energía negativa
# b_scan_processed = np.clip(b_scan_processed, 0, None)
# indices_z, indices_x = np.where(b_scan_processed > 0)
# plt.figure()
# plt.title('Drawing contours to detect possible piece of stones')
# plt.imshow(b_scan_processed, extent = [0, 5, 4, 0], aspect ='auto', cmap='inferno')
# map_subsurface = b_scan_processed
# plt.contour(X, Z, map_subsurface>0.75, levels=[0.1], colors='white')
# plt.scatter([r['x'] for r in stones], [r['z'] for r in stones], marker='+', s=120, color='cyan', label='phase center')
# plt.legend(loc='lower left')
# plt.show()


# 4  Super resolution
factor = 10
b_scan_super_res = ndimage.zoom(b_scan_migration, zoom=(factor, factor), order = 3)
noise_th = np.max(b_scan_super_res) * 0.4
b_scan_super_res = np.clip(b_scan_super_res - noise_th, 0, None)

# #new axis
x_axis_sr = np.linspace(0, 5, len(x_antenna) * factor)
z_axis_sr = np.linspace(0, depths[-1], len(depths) * factor)



############################# Krigin for peaks detection
th_atr = np.max(b_scan_super_res) * 0.7 ## remainign percertange of energy 30 %
neighbours = 15
max_val_image = ndimage.maximum_filter(b_scan_super_res, size = neighbours)

peaks_mask = (b_scan_super_res==max_val_image) & (b_scan_super_res>th_atr)
index_z_sr, index_x_sr = np.where(peaks_mask)
#X_SR, Z_SR = np.meshgrid(grid_x_hd, grid_z_hd)

X_SR, Z_SR = np.meshgrid(x_axis_sr, z_axis_sr)
detected_targets = []
for i in range(len(index_x_sr)):
    x_det = x_axis_sr[index_x_sr[i]]
    z_det = z_axis_sr[index_z_sr[i]]
    detected_targets.append((x_det, z_det))
plt.subplot(133)
plt.title('Phase center ')
plt.imshow(b_scan_super_res, extent = [0, 5, depths[-1], 0], aspect ='auto', cmap='inferno')
plt.contour(X_SR, Z_SR, b_scan_super_res, levels=3, colors='white')
for i, (x_det, z_det) in enumerate(detected_targets):
    plt.plot(x_det, z_det, 'w+', markersize = 15)
    plt.Rectangle([x_det, z_det], 0.4, 0.4,
                  linewidth=2, linestyle='-')
    plt.text(x_det +0.25, z_det, f'Tg{i + 1}', color='lime', fontweight='bold')
plt.xlabel("Position [m]")
plt.show()



vol_ += np.random.normal(0, 0.3, vol_.shape)
vol_ += np.random.uniform(0, 0.5, vol_.shape)
vol_ = np.clip(vol_, 0, None)

vol_ = ndimage.gaussian_filter(vol_, sigma=1.5)
vol_dbscan = vol_
thd = np.max(vol_) *0.6
neigh = 5

max_3d = ndimage.maximum_filter(vol_, size = neigh)
mask_peaks_3d = (vol_ == max_3d) & (vol_> thd)
inde_z, inde_y, inde_x = np.where(mask_peaks_3d)

z_plot, y_plot, x_plot = np.where(vol_>thd)
val_plot  = max_3d[z_plot, y_plot, x_plot]



fig3d= plt.figure(figsize=(11, 8))
ax3d = fig3d.add_subplot(121, projection='3d')

ax3d.scatter(x_grid[x_plot], y_grid[y_plot], -z_grid[z_plot],
            c= val_plot, alpha=0.5)
#ax3d.scatter(x_grid[inde_x], y_grid[inde_y], -z_grid[inde_z], color='lime', marker='*', s=200, edgecolor='black')
ax3d.set_xlim([0, 5])
ax3d.set_ylim([0, 4])
ax3d.set_zlim([-4, 0])
ax3d.set_title('Phase center detected on Point Cloud')
plt.show()

### KDTREE NOise

from scipy.spatial import cKDTree





### Using Density Based Spatial Clustering of Applications with Noise (DBSCAN).
from sklearn.cluster import DBSCAN
thd_vis = np.max(vol_dbscan) * 0.6
index_z_dbscan, index_y_dbscan, index_x_dbscan = np.where(vol_dbscan>thd_vis)


physic_peaks= np.vstack([x_grid[index_x_dbscan], y_grid[index_y_dbscan], z_grid[index_z_dbscan]]).T

start_time = time.time()

dbscan_alg = DBSCAN(eps= 0.6, min_samples= 30)
labels_dbscan = dbscan_alg.fit_predict(physic_peaks)

clusters= set(labels_dbscan) - {-1}
num_tg = len(clusters)

###########################################

print ("Report:")
if num_tg ==0:
    print("Diagnostic: Solid stone")
else:
    print ("DIagnostic: Broke stone")
    print("It detect a broke zone wit ", num_tg)

    for i, cluster_i in enumerate(clusters):
        grp_point = physic_peaks[labels_dbscan== cluster_i]
        center_x, center_y, center_z = np.mean(grp_point, axis = 0)
        power = len(grp_point)
        print(
            f"   -> Group #{i + 1} (Fragiel zone) -> Center: X:{center_x:.1f}m | Y:{center_y:.1f}m | Z(Prof):{center_z:.1f}m (Magnitude: {power})")

############################################
end = time.time()
print("Execution time (DBSCAN)", end - start_time)
noise_mask = (labels_dbscan == -1)
tg_mask = (labels_dbscan != -1)

#fig_3d_dbsca = plt.figure(figsize=(11, 8))
ax_3d_dbscan = fig3d.add_subplot(122, projection='3d')
ax_3d_dbscan.scatter(physic_peaks[noise_mask, 0],
                     physic_peaks[noise_mask, 1],
                     -physic_peaks[noise_mask, 2],
                     c= 'grey', alpha = 0.03, s= 5)
scater_dcscan =ax_3d_dbscan.scatter(physic_peaks[tg_mask, 0],
                     physic_peaks[tg_mask, 1],
                     -physic_peaks[tg_mask, 2],
                     c=labels_dbscan[tg_mask],
                     cmap ='tab10',
                     s=30, alpha = 0.4)

for i, cluster in enumerate(clusters):
    center_x, center_y, center_z = np.mean(physic_peaks[labels_dbscan == cluster], axis=0)
    ax_3d_dbscan.scatter(center_x, center_y, -center_z,
                         color= 'white', marker='*', s=20,
                         edgecolors='black', zorder=5)
    ax_3d_dbscan.text(center_x, center_y, -center_z + 0.3,f"Fragile zone {i+1}",
                      fontweight='bold', fontsize=11, zorder= 6)
ax_3d_dbscan.set_ylim([0, 3])
ax_3d_dbscan.set_xlim([0, 5])
ax_3d_dbscan.set_zlim([-4, 0])
ax_3d_dbscan.set_xlabel(' [m]')
ax_3d_dbscan.set_ylabel(' [m]')
ax_3d_dbscan.set_zlabel(' [m]')
ax_3d_dbscan.set_title("DBSCAN on point Cloud")