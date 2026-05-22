import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import butter, filtfilt

# ==========================================
# 1. PARÁMETROS DEL RADAR (Alta Resolución ~1.5 GHz)
# ==========================================
c = 3e8;
eps_roca = 6.0;
v_roca = c / np.sqrt(eps_roca)
f_start = 800e6;
f_end = 2.2e9;
B = f_end - f_start;
T_sweep = 2e-6
fs_rf = 6e9;
t = np.arange(0, T_sweep, 1 / fs_rf);
N = len(t)

TX_Chirp = np.exp(2 * np.pi * (f_start * t + (B / (2 * T_sweep)) * t ** 2))
b_filter, a_filter = butter(4, 30e6 / (0.5 * fs_rf), btype='low')

# ==========================================
# 2. ESCENARIO: Tres rocas esféricas de diferentes tamaños
# ==========================================
x_antena = np.linspace(0, 4, 80)
rocas = [
    {"x": 1.0, "z": 0.8, "radio": 0.15, "fuerza": 1.0},  # Roca superficial
    {"x": 2.5, "z": 1.5, "radio": 0.25, "fuerza": 1.2},  # Roca profunda grande
    {"x": 3.2, "z": 0.6, "radio": 0.10, "fuerza": 0.8}  # Piedra pequeña
]

# ==========================================
# 3. PASO 1: SIMULACIÓN DEL RADAR CRUDO (Hipérbolas)
# ==========================================
b_scan_raw = np.zeros((N // 2, len(x_antena)))
print("1. Capturando datos crudos (Formando hipérbolas)...")

for i, x_ant in enumerate(x_antena):
    RX_Total = np.zeros_like(t)
    for roca in rocas:
        # El eco rebotará principalmente en el TECHO de la roca (z - radio)
        techo_z = roca["z"] - roca["radio"]
        dist = np.sqrt((x_ant - roca["x"]) ** 2 + techo_z ** 2)
        tau = 2 * dist / v_roca
        t_rx = t - tau
        valid = t_rx >= 0

        fase_rx = np.zeros_like(t)
        fase_rx[valid] = 2 * np.pi * (f_start * t_rx[valid] + (B / (2 * T_sweep)) * t_rx[valid] ** 2)
        RX_Total[valid] += (roca["fuerza"] / dist) * np.cos(fase_rx[valid])

    RX_Total += np.random.normal(0, 0.05, N)  # Ruido
    Beat_Signal = filtfilt(b_filter, a_filter, TX_Chirp * RX_Total)
    b_scan_raw[:, i] = np.abs(np.fft.fft(Beat_Signal))[:N // 2]

# Mapeo a Profundidad
frecuencias = np.fft.fftfreq(N, 1 / fs_rf)[:N // 2]
profundidades = frecuencias * T_sweep * v_roca / (2 * B)
idx_max = np.argmax(profundidades > 2.5)
b_scan_raw = b_scan_raw[:idx_max, :]
profundidades = profundidades[:idx_max]

# ==========================================
# 4. PASO 2: MIGRACIÓN MATEMÁTICA (Puntos Brillantes)
# ==========================================
print("2. Aplicando Migración (Concentrando en puntos brillantes)...")
b_scan_migrado = np.zeros_like(b_scan_raw)

for i, x_out in enumerate(x_antena):
    for j, z_out in enumerate(profundidades):
        if z_out == 0: continue
        dist_hip = np.sqrt(z_out ** 2 + (x_antena - x_out) ** 2)
        idx_z_busc = np.searchsorted(profundidades, dist_hip)
        validos = idx_z_busc < len(profundidades)
        if np.any(validos):
            b_scan_migrado[j, i] = np.sum(b_scan_raw[idx_z_busc[validos], np.arange(len(x_antena))[validos]])

# ==========================================
# 5. VISUALIZACIÓN: LA INTERFAZ DEL SOFTWARE
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
extent = [0, 4, 2.5, 0]

# Panel 1: Lo que capta la antena
axes[0].imshow(b_scan_raw, extent=extent, aspect='auto', cmap='gray')
axes[0].set_title('1. Datos Crudos\n(Lo que capta el hardware)')
axes[0].set_ylabel('Profundidad (m)')
axes[0].set_xlabel('Posición (m)')

# Panel 2: Lo que hace la física
axes[1].imshow(b_scan_migrado, extent=extent, aspect='auto', cmap='magma')
axes[1].set_title('2. Procesamiento Geofísico\n(El "Punto Brillante")')
axes[1].set_xlabel('Posición (m)')

# Panel 3: LO QUE TÚ LE MUESTRAS AL CLIENTE
axes[2].imshow(b_scan_migrado, extent=extent, aspect='auto', cmap='magma')
axes[2].set_title('3. Tu Software de Interpretación\n(Dibujando la geometría)')
axes[2].set_xlabel('Posición (m)')

# Aquí simulamos el algoritmo de tu software dibujando las formas sobre los puntos
for idx, roca in enumerate(rocas):
    # El software detecta el punto brillante y dibuja un círculo estimado
    # Notar que el centro geométrico de la roca está un poco más abajo que el techo brillante
    circulo = patches.Circle((roca["x"], roca["z"]), radius=roca["radio"],
                             linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
    axes[2].add_patch(circulo)

    # Etiqueta automática del software
    axes[2].text(roca["x"] + 0.15, roca["z"], f'Objeto {idx + 1}\nProf: {roca["z"]}m',
                 color='cyan', fontsize=10, fontweight='bold',
                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

plt.tight_layout()
plt.show()