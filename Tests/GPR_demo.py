import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ==========================================
# 1. PARÁMETROS DEL HARDWARE RF (Baja Frecuencia)
# ==========================================
c = 3e8
eps_roca = 6.0
v_roca = c / np.sqrt(eps_roca)

# Georradar de baja frecuencia para penetración profunda
f_start = 500e6  # 50 MHz
f_end = 2.0e9  # 250 MHz
B = f_end - f_start  # Ancho de banda: 200 MHz
T_sweep = 5e-6

# Muestreo a 1 GHz (Suficiente para digitalizar 250 MHz sin aliasing)
fs_rf = 2e9
t = np.arange(0, T_sweep, 1 / fs_rf)
N = len(t)

# Generar TX y Filtro Pasabajos
TX_Chirp = np.cos(2 * np.pi * (f_start * t + (B / (2 * T_sweep)) * t ** 2))
b_filter, a_filter = butter(4, 15e6 / (0.5 * fs_rf), btype='low')

# ==========================================
# 2. EL ENTORNO (Aire + Pared + Rocas rotas)
# ==========================================
# Ampliamos la profundidad a 4 metros para ver el efecto de la baja frecuencia
x_grid = np.linspace(0, 5, 120)
z_grid = np.linspace(0, 4, 100)
X, Z = np.meshgrid(x_grid, z_grid)
mapa_subsuelo = np.zeros_like(X)

z_pared = 0.5  # La pared de la mina empieza a medio metro de la antena

# 1. LA PARED (Capa principal frontal)
mapa_subsuelo[(Z >= z_pared) & (Z <= z_pared + 0.1)] = 0.8

rocas_generadas = []
# 2. ZONA ESCARBADA (Piedras rotas/escombros esparcidos)
# En lugar de un círculo perfecto, generamos fragmentos de diferentes tamaños
np.random.seed(15)  # Para que la "rotura" sea igual cada vez que corras el código
for _ in range(25):  # 25 pedazos de roca
    rx = np.random.uniform(2.5, 4.5)  # Repartidos a la derecha
    rz = np.random.uniform(1.2, 3.5)  # Detrás de la pared
    r_size = np.random.uniform(0.01, 0.08)  # Diferentes tamaños (piedras pequeñas y grandes)
    mapa_subsuelo[(X - rx) ** 2 + (Z - rz) ** 2 < r_size] = np.random.uniform(0.4, 0.9)
    rocas_generadas.append((rx, rz))
# 3. UN BLOQUE INTACTO (Para contrastar)
bx, bz = 1.5, 1.0
mapa_subsuelo[(X - bx) ** 2 + (Z - bz) ** 2 < 0.05] = 1.0

indices_z, indices_x = np.where(mapa_subsuelo > 0)

# ==========================================
# 3. EL ESCANEO CIEGO DEL RADAR (Con 2 velocidades)
# ==========================================
x_antena = np.linspace(0, 5, 100)
b_scan_raw = np.zeros((N // 2, len(x_antena)))
print("Radar escaneando... (Considerando cambio de velocidad en la pared)")
for i, x_ant in enumerate(x_antena):
    RX_Total = np.zeros_like(t)

    for idx in range(len(indices_x)):
        x_pixel = x_grid[indices_x[idx]]
        z_pixel = z_grid[indices_z[idx]]
        reflectividad = mapa_subsuelo[indices_z[idx], indices_x[idx]]

        # FÍSICA DE 2 MEDIOS: Aire y Roca
        dist = np.sqrt((x_ant - x_pixel) ** 2 + z_pixel ** 2)

        if z_pixel <= z_pared:
            # Si el objeto está en el aire (ej. la pared misma)
            tau = 2 * dist / c
        else:
            # Triángulos semejantes para saber cuánto viajó en aire y cuánto en roca
            L_aire = dist * (z_pared / z_pixel)
            L_roca = dist - L_aire
            tau = 2 * (L_aire / c + L_roca / v_roca)

        t_rx = t - tau
        valid = t_rx >= 0
        fase_rx = np.zeros_like(t)
        fase_rx[valid] = 2 * np.pi * (f_start * t_rx[valid] + (B / (2 * T_sweep)) * t_rx[valid] ** 2)

        RX_Total[valid] += (reflectividad / (dist ** 1.2)) * np.cos(fase_rx[valid])

    # Ruido eléctrico realista
    RX_Total += np.random.normal(0, 0.05, N)

    Mixed_Signal = TX_Chirp * RX_Total
    Beat_Signal = filtfilt(b_filter, a_filter, Mixed_Signal)
    b_scan_raw[:, i] = np.abs(np.fft.fft(Beat_Signal))[:N // 2]

# ==========================================
# 4. MIGRACIÓN Y CONVERSIÓN DE PROFUNDIDAD
# ==========================================
frecuencias = np.fft.fftfreq(N, 1 / fs_rf)[:N // 2]
tiempos_vuelo = frecuencias * T_sweep / B

# Mapeo no lineal de profundidad (porque hay aire primero, luego roca)
profundidades = np.zeros_like(tiempos_vuelo)
tau_pared = 2 * z_pared / c

for k, tau in enumerate(tiempos_vuelo):
    if tau <= tau_pared:
        profundidades[k] = tau * c / 2
    else:
        profundidades[k] = z_pared + (tau - tau_pared) * v_roca / 2

# Recortar la vista hasta 4 metros
idx_max = np.argmax(profundidades > 4.0)
b_scan_raw = b_scan_raw[:idx_max, :]
profundidades = profundidades[:idx_max]
tiempos_vuelo = tiempos_vuelo[:idx_max]

print("Aplicando Migración adaptada a la pared...")
b_scan_migrado = np.zeros_like(b_scan_raw)
for i, x_out in enumerate(x_antena):
    for j, z_out in enumerate(profundidades):
        if z_out == 0: continue

        dist_plano = np.abs(x_antena - x_out)
        dist_hip = np.sqrt(dist_plano ** 2 + z_out ** 2)

        # Simulamos la misma física de 2 medios para corregir la curva
        tiempos_curva = np.zeros_like(dist_hip)
        for k_idx, d in enumerate(dist_hip):
            if z_out <= z_pared:
                tiempos_curva[k_idx] = 2 * d / c
            else:
                L_aire = d * (z_pared / z_out)
                L_roca = d - L_aire
                tiempos_curva[k_idx] = 2 * (L_aire / c + L_roca / v_roca)

        # Buscar el índice temporal más cercano
        for k_idx, t_c in enumerate(tiempos_curva):
            idx_t = np.argmin(np.abs(tiempos_vuelo - t_c))
            if idx_t < len(tiempos_vuelo):
                b_scan_migrado[j, i] += b_scan_raw[idx_t, k_idx]

# ==========================================
# 5. GRÁFICOS
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
extent = [0, 5, 4, 0]

axes[0].imshow(mapa_subsuelo, extent=extent, aspect='auto', cmap='binary')
axes[0].set_title('1. La Realidad Oculta\n(Aire + Pared + Rocas rotas)')
axes[0].set_ylabel('Profundidad (m)')
axes[0].axhline(y=z_pared, color='blue', linestyle='--', alpha=0.5, label='Inicio Pared')
axes[0].legend()

axes[1].imshow(b_scan_raw, extent=extent, aspect='auto', cmap='viridis')
axes[1].set_title('2. Radargrama Crudo a 200 MHz\n(Baja resolución)')

axes[2].imshow(b_scan_migrado, extent=extent, aspect='auto', cmap='inferno')
axes[2].set_title('3. Migración Final')

# Dibujamos el contorno exacto de las rocas originales sobre la mancha de energía
# Usamos 'contour' para dibujar las siluetas del 'mapa_subsuelo' en color blanco
axes[2].contour(X, Z, mapa_subsuelo, levels=[0.1], colors='white', linewidths=1.5, alpha=0.6)

# Agregamos marcadores específicos para guiar el ojo
axes[2].axhline(y=z_pared, color='cyan', linestyle='--', linewidth=2, alpha=0.8, label='Línea de Pared detectada')
axes[2].scatter(bx, bz, edgecolor='lime', facecolor='none', s=250, linewidth=2, label='Bloque Intacto (Detectado)')

# Marcamos el área de escombros
axes[2].scatter(rocas_generadas[0][0], rocas_generadas[0][1], color='cyan', marker='x', label='Zona de Escombros')
for rx, rz in rocas_generadas[1:]:
    axes[2].scatter(rx, rz, color='cyan', marker='x', alpha=0.4)

axes[2].legend(loc='lower left', framealpha=0.9)

plt.tight_layout()
plt.show()
plt.tight_layout()
plt.show()