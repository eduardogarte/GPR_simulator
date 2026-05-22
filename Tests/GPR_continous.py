import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ==========================================
# 1. PARÁMETROS DEL HARDWARE RF (FMCW)
# ==========================================
c = 3e8
eps_roca = 6.0
v_roca = c / np.sqrt(eps_roca)

f_start = 0;
f_end = 200e6
B = f_end - f_start;
T_sweep = 5e-6
fs_rf = 400e6
t = np.arange(0, T_sweep, 1 / fs_rf)
N = len(t)

# Generar TX y Filtro Pasabajos
TX_Chirp = np.cos(2 * np.pi * (f_start * t + (B / (2 * T_sweep)) * t ** 2))
b_filter, a_filter = butter(4, 25e6 / (0.5 * fs_rf), btype='low')


plt.figure()
plt.plot(TX_Chirp)
# ==========================================
# 2. EL ENTORNO "IN SITU" (La Caja Negra)
# ==========================================
# En lugar de objetos discretos, creamos una cuadrícula 2D (Resolución de 5cm)
x_grid = np.linspace(0, 5, 100)
z_grid = np.linspace(0, 2.5, 50)
X, Z = np.meshgrid(x_grid, z_grid)

# Matriz de reflectividad del suelo (0 = roca sólida sana, no hay rebote)
mapa_subsuelo = np.zeros_like(X)

# "Pintamos" defectos en el mapa sin que el radar lo sepa
# 1. Una delaminación (grieta plana) entre x=1 y x=2, a 0.8m de profundidad
mapa_subsuelo[(X > 1.0) & (X < 2.0) & (Z >= 0.8) & (Z <= 0.85)] = 0.6

# 2. Un bloque de roca roto y amorfo (ruido/caos) cerca de x=3.5, z=1.5
mapa_subsuelo[(X - 3.5) ** 2 + (Z - 1.5) ** 2 < 0.08] = 0.9

# Para optimizar el cálculo en Python, extraemos solo los píxeles que rebotan
# (En una simulación FDTD completa, la onda cruza los píxeles vacíos también)
indices_z, indices_x = np.where(mapa_subsuelo > 0)

# ==========================================
# 3. EL ESCANEO CIEGO DEL RADAR
# ==========================================
x_antena = np.linspace(0, 5, 100)
b_scan_raw = np.zeros((N // 2, len(x_antena)))

print("Radar escaneando la cuadrícula a ciegas...")
for i, x_ant in enumerate(x_antena):
    RX_Total = np.zeros_like(t)

    # El radar interactúa con TODOS los puntos del espacio que tienen un cambio
    for idx in range(len(indices_x)):
        x_pixel = x_grid[indices_x[idx]]
        z_pixel = z_grid[indices_z[idx]]
        reflectividad = mapa_subsuelo[indices_z[idx], indices_x[idx]]

        # Física básica de viaje
        dist = np.sqrt((x_ant - x_pixel) ** 2 + z_pixel ** 2)
        tau = 2 * dist / v_roca

        t_rx = t - tau
        valid = t_rx >= 0
        fase_rx = np.zeros_like(t)
        fase_rx[valid] = 2 * np.pi * (f_start * t_rx[valid] + (B / (2 * T_sweep)) * t_rx[valid] ** 2)

        # Sumamos el eco de este píxel
        RX_Total[valid] += (reflectividad / (dist ** 1.5)) * np.cos(fase_rx[valid])

    RX_Total += np.random.normal(0, 15, N)  # Ruido eléctrico

    # Hardware RF a Banda Base
    Mixed_Signal = TX_Chirp * RX_Total
    Beat_Signal = filtfilt(b_filter, a_filter, Mixed_Signal)
    b_scan_raw[:, i] = np.abs(np.fft.fft(Beat_Signal))[:N // 2]

# ==========================================
# 4. MIGRACIÓN Y VISUALIZACIÓN
# ==========================================
frecuencias = np.fft.fftfreq(N, 1 / fs_rf)[:N // 2]
profundidades = frecuencias * T_sweep * v_roca / (2 * B)

idx_max = np.argmax(profundidades > 2.5)
b_scan_raw = b_scan_raw[:idx_max, :]
profundidades = profundidades[:idx_max]

# Migración de Kirchhoff
print("Aplicando Migración...")
b_scan_migrado = np.zeros_like(b_scan_raw)
for i, x_out in enumerate(x_antena):
    for j, z_out in enumerate(profundidades):
        if z_out == 0: continue
        dist_curva = np.sqrt(z_out ** 2 + (x_antena - x_out) ** 2)
        indices_z_busc = np.searchsorted(profundidades, dist_curva)
        validos = indices_z_busc < len(profundidades)
        if np.any(validos):
            b_scan_migrado[j, i] = np.sum(b_scan_raw[indices_z_busc[validos], np.arange(len(x_antena))[validos]])

# --- Gráficos ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
extent = [0, 5, 2.5, 0]

axes[0].imshow(mapa_subsuelo, extent=extent, aspect='auto', cmap='binary')
axes[0].set_title('1. La Realidad Oculta (Matriz)')
axes[0].set_ylabel('Profundidad (m)')

axes[1].imshow(b_scan_raw, extent=extent, aspect='auto', cmap='viridis')
axes[1].set_title('2. Lo que el hardware mide (Crudo)')

axes[2].imshow(b_scan_migrado, extent=extent, aspect='auto', cmap='inferno')
axes[2].set_title('3. El procesamiento final (Migrado)')

plt.tight_layout()
plt.show()