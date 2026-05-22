import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ==========================================
# 1. PARÁMETROS DEL MEDIO Y DEL HARDWARE RF (Capa Física)
# ==========================================
c = 3e8
eps_roca = 6.0
v_roca = c / np.sqrt(eps_roca)  # Velocidad: ~1.22e8 m/s

# Diseño del Oscilador (VCO) para generar el Chirp
f_start = 0.5e9  # 500 MHz
f_end = 1.5e9  # 1.5 GHz
B = f_end - f_start
T_sweep = 5e-6  # Barrido de 5 microsegundos

# Muestreo de RF (Simulación a 4 GHz para modelar la realidad analógica)
fs_rf = 400e6
t = np.arange(0, T_sweep, 1 / fs_rf)
N = len(t)

# Generación de la onda TX maestra (Microondas puras)
fase_tx = 2 * np.pi * (f_start * t + (B / (2 * T_sweep)) * t ** 2)
TX_Chirp = np.cos(fase_tx)

plt.figure()
plt.plot(t, TX_Chirp)
# Filtro Pasabajos Analógico (El que deja pasar solo la Banda Base)
nyq = 0.5 * fs_rf
fc_lp = 25e6  # Todo lo que esté por encima de 25 MHz (incluyendo los 2 GHz) se borra
b_filter, a_filter = butter(16, fc_lp / nyq, btype='low')

# ==========================================
# 2. EL ESCENARIO (Escaneo en el Techo de la Mina)
# ==========================================
x_antena = np.linspace(0, 5, 100)  # Movemos el radar en 100 pasos (5 metros)
rocas_rotas = [
    {"x": 1.5, "z": 0.8, "fuerza": 1.0},
    {"x": 2.5, "z": 1.5, "fuerza": 0.8},
    {"x": 3.5, "z": 1.1, "fuerza": 0.9},
    {"x": 3.0, "z": 1.0, "fuerza": 1.0},
    {"x": 2.0, "z": 4.5, "fuerza": 0.8}
]

# Matriz para la imagen generada por el hardware
b_scan_raw = np.zeros((N // 2, len(x_antena)))

print("1. El Hardware está escaneando: Multiplicando GHz y filtrando a Banda Base...")

# ==========================================
# 3. EL PIPELINE ELECTRÓNICO (En cada paso de la antena)
# ==========================================
for i, x in enumerate(x_antena):
    RX_Total = np.zeros_like(t)

    # a. Física: Las ondas rebotan en las rocas y regresan a la antena
    for roca in rocas_rotas:
        dist = np.sqrt((x - roca["x"]) ** 2 + roca["z"] ** 2)
        tau = 2 * dist / v_roca

        t_rx = t - tau
        valid = t_rx >= 0

        fase_rx = np.zeros_like(t)
        fase_rx[valid] = 2 * np.pi * (f_start * t_rx[valid] + (B / (2 * T_sweep)) * t_rx[valid] ** 2)

        amp = roca["fuerza"] / (dist ** 1.5)
        RX_Total[valid] += amp * np.cos(fase_rx[valid])

    RX_Total += np.random.normal(0, 0.5 , N)  # Ruido eléctrico

    # b. Hardware: MEZCLADOR ANALÓGICO (Downconversion)
    Mixed_Signal = TX_Chirp * RX_Total

    # c. Hardware: FILTRO PASABAJOS (Aísla la Banda Base)
    Beat_Signal = filtfilt(b_filter, a_filter, Mixed_Signal)

    # d. Software Base: El microcontrolador lee la Banda Base y aplica FFT
    b_scan_raw[:, i] = np.abs(np.fft.rfft(Beat_Signal))[:N // 2]



plt.figure()
plt.plot(t, Mixed_Signal)
# ==========================================
# 4. MAPEO: De Frecuencia a Profundidad Real
# ==========================================
frecuencias = np.fft.fftfreq(N, 1 / fs_rf)[:N // 2]
profundidades = frecuencias * T_sweep * v_roca / (2 * B)

# Recortamos la imagen hasta 2.5 metros de profundidad para ver los objetivos
idx_max = np.argmax(profundidades > 6.5)
b_scan_raw = b_scan_raw[:idx_max, :]
profundidades = profundidades[:idx_max]

# ==========================================
# 5. EL SOFTWARE AVANZADO: MIGRACIÓN DE KIRCHHOFF
# ==========================================
print("2. El Software está procesando: Colapsando hipérbolas con Migración...")
b_scan_migrado = np.zeros_like(b_scan_raw)

# El algoritmo barre el radargrama concentrando las "curvas" en puntos
for i, x_out in enumerate(x_antena):
    for j, z_out in enumerate(profundidades):
        if z_out == 0: continue

        # Ecuación teórica de cómo se vería una hipérbola en este punto
        dist_curva = np.sqrt(z_out ** 2 + (x_antena - x_out) ** 2)

        # Buscamos esos índices en el radargrama generado por el hardware
        indices_z = np.searchsorted(profundidades, dist_curva)
        validos = indices_z < len(profundidades)

        if np.any(validos):
            # Sumamos toda la energía y la guardamos en el punto focal
            energia = np.sum(b_scan_raw[indices_z[validos], np.arange(len(x_antena))[validos]])
            b_scan_migrado[j, i] = energia

# ==========================================
# 6. VISUALIZACIÓN EN PANTALLA
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
extent = [x_antena[0], x_antena[-1], profundidades[-1], profundidades[0]]

# Gráfico 1: Lo que entrega el circuito
im1 = axes[0].imshow(b_scan_raw, extent=extent, aspect='auto', cmap='viridis')
axes[0].set_title('1. Radargrama Crudo (Ondas llevadas a Banda Base por el Mezclador)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Profundidad en roca (m)')
axes[0].scatter([r['x'] for r in rocas_rotas], [r['z'] for r in rocas_rotas], color='red', marker='x', s=100,
                label='Peligro Real Oculto')
axes[0].legend(loc='lower left')

# Gráfico 2: Lo que entrega el procesamiento digital
im2 = axes[1].imshow(b_scan_migrado, extent=extent, aspect='auto', cmap='inferno')
axes[1].set_title('2. Radargrama Migrado (Energía enfocada matemáticamente)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Posición en el techo de la mina (m)')
axes[1].set_ylabel('Profundidad en roca (m)')
axes[1].scatter([r['x'] for r in rocas_rotas], [r['z'] for r in rocas_rotas], color='cyan', marker='+', s=120,
                label='Peligro Real Oculto')
axes[1].legend(loc='lower left')

plt.tight_layout()
plt.show()