import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ==========================================
# 1. PARÁMETROS BÁSICOS Y HARDWARE RF
# ==========================================
c = 3e8;
eps_roca = 6.0;
v_roca = c / np.sqrt(eps_roca)
f_start = 0;
f_end = 500e6;
B = f_end - f_start;
T_sweep = 5e-6
fs_rf = 4e9;
t = np.arange(0, T_sweep, 1 / fs_rf);
N = len(t)

# Transmisor y Filtro
TX_Chirp = np.cos(2 * np.pi * (f_start * t + (B / (2 * T_sweep)) * t ** 2))
b_filter, a_filter = butter(4, 25e6 / (0.5 * fs_rf), btype='low')

# ==========================================
# 2. EL ESCENARIO REALISTA (Peligros + Ruido Ambiental)
# ==========================================
x_antena = np.linspace(0, 5, 100)

# A) Los Peligros Reales (Rocas grandes sueltas que queremos detectar)
peligros = [
    {"x": 2.5, "z": 1.2, "fuerza": 1.0},  # Peligro central
    {"x": 1.0, "z": 0.8, "fuerza": 0.9}  # Peligro lateral
]

# B) GENERACIÓN DE RUIDO AMBIENTAL NATURAL (Geological Clutter)
# Simulamos 300 pequeñas impurezas, humedad y microgrietas esparcidas por toda la roca
np.random.seed(42)  # Para reproducibilidad
num_clutter = 300
clutter_x = np.random.uniform(0, 5, num_clutter)
clutter_z = np.random.uniform(0.1, 4.5, num_clutter)
clutter_fuerza = np.random.uniform(0.01, 0.8, num_clutter)  # Ecos muy débiles

b_scan_crudo = np.zeros((N // 2, len(x_antena)))

print("Escaneando con ruido eléctrico y ambiental (Clutter)...")
for i, x in enumerate(x_antena):
    RX_Total = np.zeros_like(t)

    # --- 1. Ecos de los Peligros Reales ---
    for p in peligros:
        dist = np.sqrt((x - p["x"]) ** 2 + p["z"] ** 2)
        tau = 2 * dist / v_roca
        t_rx = t - tau;
        valid = t_rx >= 0
        fase_rx = np.zeros_like(t)
        fase_rx[valid] = 2 * np.pi * (f_start * t_rx[valid] + (B / (2 * T_sweep)) * t_rx[valid] ** 2)
        RX_Total[valid] += (p["fuerza"] / (dist ** 1.2)) * np.cos(fase_rx[valid])

    # --- 2. RUIDO AMBIENTAL: Ecos del Clutter Geológico (La "niebla" de la roca) ---
    for cx, cz, camp in zip(clutter_x, clutter_z, clutter_fuerza):
        dist = np.sqrt((x - cx) ** 2 + cz ** 2)
        tau = 2 * dist / v_roca
        t_rx = t - tau;
        valid = t_rx >= 0
        fase_rx = np.zeros_like(t)
        fase_rx[valid] = 2 * np.pi * (f_start * t_rx[valid] + (B / (2 * T_sweep)) * t_rx[valid] ** 2)
        RX_Total[valid] += (camp / (dist ** 1.2)) * np.cos(fase_rx[valid])

    # --- 3. RUIDO AMBIENTAL: Rugosidad del techo (Acople de antena variable) ---
    # La antena rebota un poco al arrastrarla, cambiando el eco directo de la superficie
    ruido_rugosidad = np.random.uniform(0.1, 0.3)
    RX_Total += ruido_rugosidad * TX_Chirp  # Eco instantáneo desordenado

    # --- 4. RUIDO ELÉCTRICO: Interferencia de maquinaria y térmica del hardware ---
    RX_Total += np.random.normal(0, 0.06, N)

    # Procesamiento Hardware -> Software
    Beat_Signal = filtfilt(b_filter, a_filter, TX_Chirp * RX_Total)
    b_scan_crudo[:, i] = np.abs(np.fft.fft(Beat_Signal))[:N // 2]

# Mapeo a Profundidad
frecuencias = np.fft.fftfreq(N, 1 / fs_rf)[:N // 2]
profundidades = frecuencias * T_sweep * v_roca / (2 * B)
idx_max = np.argmax(profundidades > 2.5)

b_scan_raw = b_scan_crudo[:idx_max, :]
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


# Visualización
plt.figure(figsize=(10, 6))
plt.imshow(b_scan_crudo[:idx_max, :], extent=[0, 5, 2.5, 0], aspect='auto', cmap='viridis')
plt.title('Radargrama con Ruido Ambiental (Clutter Geológico) y Eléctrico', fontsize=14)
plt.xlabel('Posición (m)');
plt.ylabel('Profundidad (m)')
plt.colorbar(label='Amplitud del Eco')


plt.figure(figsize=(10, 6))
plt.imshow(b_scan_migrado, extent=[0, 5, 2.5, 0], aspect='auto', cmap='inferno')
plt.title('MIgration')

# Marcar dónde están realmente los peligros mayores
for p in peligros:
    plt.plot(p["x"], p["z"], 'r+', markersize=15, markeredgewidth=2, label='Peligro Real')

# Evitar que la leyenda se repita
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()