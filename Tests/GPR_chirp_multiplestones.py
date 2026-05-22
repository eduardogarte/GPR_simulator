import numpy as np
import matplotlib.pyplot as plt

# --- 1. PARÁMETROS DEL RADAR FMCW Y LA ROCA ---
c = 3e8
eps_roca = 6.0
v_roca = c / np.sqrt(eps_roca)  # Velocidad en la roca: ~1.22e8 m/s

B = 1.5e9  # Ancho de banda: 1.5 GHz (Alta resolución)
T_sweep = 10e-6  # Barrido de 10 microsegundos
fs_adc = 1e9  # Muestreo ADC de 40 MHz
t_adc = np.arange(0, T_sweep, 1 / fs_adc)
N = len(t_adc)

# Escaneo en el techo de la mina (5 metros)
x_antena = np.linspace(0, 5, 150)

# --- 2. ESCENARIO REALISTA: Múltiples rocas fracturadas por voladura ---
# Cada diccionario es un bloque de roca roto oculto en el techo
rocas_rotas = [
    {"x": 1.5, "z": 0.8, "fuerza": 1.0},  # Roca superficial izquierda
    {"x": 2.2, "z": 1.5, "fuerza": 0.8},  # Roca profunda central (peligrosa)
    {"x": 3.0, "z": 1.1, "fuerza": 0.9},  # Roca media derecha
    {"x": 3.8, "z": 0.6, "fuerza": 1.0}  # Roca muy superficial derecha
]

# Matriz para el Radargrama Crudo
b_scan_raw = np.zeros((N // 2, len(x_antena)))

print("1. Escaneando techo y generando Radargrama FMCW Crudo...")
for i, x in enumerate(x_antena):
    senal_batido_total = np.zeros(N)

    for roca in rocas_rotas:
        # Distancia real de la antena a esta roca
        distancia = np.sqrt((x - roca["x"]) ** 2 + roca["z"] ** 2)
        tiempo_vuelo = 2 * distancia / v_roca

        # Frecuencia de batido para esta distancia
        fb = (B / T_sweep) * tiempo_vuelo

        # Atenuación geométrica
        amplitud = roca["fuerza"] / (distancia ** 1.5)
        senal_batido_total += amplitud * np.cos(2 * np.pi * fb * t_adc)

    # Añadimos ruido del entorno minero
    senal_batido_total += np.random.normal(0, 0.08, N)

    # FFT para obtener la profundidad
    espectro_fft = np.abs(np.fft.fft(senal_batido_total))[:N // 2]
    b_scan_raw[:, i] = espectro_fft

# Conversión de Eje a Profundidad Real (Metros)
frecuencias = np.fft.fftfreq(N, 1 / fs_adc)[:N // 2]
profundidades = frecuencias * T_sweep * v_roca / (2 * B)

# Recortamos la matriz para no mostrar hasta el fondo infinito (solo hasta 2.5m)
idx_max = np.argmax(profundidades > 2.5)
b_scan_raw = b_scan_raw[:idx_max, :]
profundidades = profundidades[:idx_max]

# --- 3. EL FILTRO MÁGICO: MIGRACIÓN DE KIRCHHOFF ---
print("2. Aplicando algoritmo de Migración (Colapsando hipérbolas)...")
b_scan_migrado = np.zeros_like(b_scan_raw)

# La migración reconstruye la imagen sumando la energía a lo largo de las curvas teóricas
for i, x_out in enumerate(x_antena):
    for j, z_out in enumerate(profundidades):
        if z_out == 0: continue

        # Calculamos la hipérbola teórica para un posible objeto en (x_out, z_out)
        dist_curva = np.sqrt(z_out ** 2 + (x_antena - x_out) ** 2)

        # Buscamos en qué píxeles (índices Z) del radargrama crudo cayó esa curva
        indices_z = np.searchsorted(profundidades, dist_curva)

        # Filtramos los que se salen de la imagen
        validos = indices_z < len(profundidades)

        # Sumamos toda la energía a lo largo de esa curva y la ponemos en el punto (x_out, z_out)
        if np.any(validos):
            energia_concentrada = np.sum(b_scan_raw[indices_z[validos], np.arange(len(x_antena))[validos]])
            b_scan_migrado[j, i] = energia_concentrada

# --- 4. VISUALIZACIÓN COMPARATIVA ---
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Parámetros visuales comunes
extent = [x_antena[0], x_antena[-1], profundidades[-1], profundidades[0]]

# Plot 1: Radargrama Crudo (El dolor de cabeza del minero)
im1 = axes[0].imshow(b_scan_raw, extent=extent, aspect='auto', cmap='viridis')
axes[0].set_title('1. Radargrama Crudo: Múltiples rocas fracturadas (Hipérbolas Solapadas)', fontsize=14,
                  fontweight='bold')
axes[0].set_ylabel('Profundidad en roca (m)')
axes[0].scatter([r['x'] for r in rocas_rotas], [r['z'] for r in rocas_rotas], color='red', marker='x', s=100,
                label='Ubicación real del daño')
axes[0].legend(loc='lower left')

# Plot 2: Radargrama Migrado (La decisión clara)
im2 = axes[1].imshow(b_scan_migrado, extent=extent, aspect='auto', cmap='inferno')
axes[1].set_title('2. Radargrama Migrado: Filtro aplicado (Identificación exacta del peligro)', fontsize=14,
                  fontweight='bold')
axes[1].set_xlabel('Posición en el techo de la mina (m)')
axes[1].set_ylabel('Profundidad en roca (m)')
axes[1].scatter([r['x'] for r in rocas_rotas], [r['z'] for r in rocas_rotas], color='cyan', marker='+', s=120,
                label='Ubicación real del daño')
axes[1].legend(loc='lower left')

plt.tight_layout()
plt.show()