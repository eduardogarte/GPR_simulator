import numpy as np
import matplotlib.pyplot as plt

# --- 1. PARÁMETROS DEL SISTEMA ---
c = 3e8  # Velocidad de la luz (m/s)
eps_roca = 6.0
v_roca = c / np.sqrt(eps_roca)  # ~1.22e8 m/s

# Parámetros del Chirp
B = 500e6  # Ancho de banda de 1 GHz (ej. 500 MHz a 1.5 GHz)
T_sweep = 10e-6  # Barrido de 10 microsegundos

# Escaneo espacial
x_antena = np.linspace(0, 5, 150)  # Movemos el radar 5 metros
# Objetivo: Un bloque de roca suelto (hueco) en x=2.5m, profundidad=1.2m
x_objetivo = 2.5
z_objetivo = 1.2

# --- 2. EL TRUCO DEL HARDWARE (Muestreo de baja frecuencia) ---
# Como vimos, el hardware ya hizo la mezcla. Solo digitalizamos el batido.
# Usamos un ADC (Convertidor Analógico-Digital) barato de 20 MHz
fs_adc = 20e6
t_adc = np.arange(0, T_sweep, 1 / fs_adc)
N = len(t_adc)

# Matriz para guardar el Radargrama final
b_scan_fmcw = np.zeros((N // 2, len(x_antena)))

print("Simulando escaneo GPR FMCW...")
# --- 3. SIMULACIÓN DEL RECORRIDO ---
for i, x in enumerate(x_antena):
    # Distancia real de la antena al objetivo en este paso (Hipotenusa)
    distancia = np.sqrt((x - x_objetivo) ** 2 + z_objetivo ** 2)
    tiempo_vuelo = 2 * distancia / v_roca

    # ¿Qué frecuencia de batido genera esta distancia físicamente?
    fb = (B / T_sweep) * tiempo_vuelo

    # Simulamos la señal digitalizada por el microcontrolador (onda de baja frecuencia)
    # Atenuamos la señal basada en la distancia (ley de la inversa del cuadrado)
    amplitud = 1.0 / (distancia ** 2)
    senal_batido = amplitud * np.cos(2 * np.pi * fb * t_adc)

    # Añadimos ruido blanco del entorno y del hardware
    senal_batido += np.random.normal(0, 0.5, N)

    # --- PROCESAMIENTO: La FFT se convierte en nuestro A-scan (traza vertical) ---
    espectro_fft = np.abs(np.fft.fft(senal_batido))[:N // 2]

    # Guardamos la columna en la imagen
    b_scan_fmcw[:, i] = espectro_fft

# --- 4. CONVERSIÓN DE FRECUENCIA A PROFUNDIDAD PARA LA IMAGEN ---
frecuencias = np.fft.fftfreq(N, 1 / fs_adc)[:N // 2]
# Ecuación del radar FMCW: d = f * T_sweep * v / (2 * B)
profundidades = frecuencias * T_sweep * v_roca / (2 * B)

# --- 5. VISUALIZACIÓN DEL RADARGRAMA ---
plt.figure(figsize=(10, 6))

# Filtramos la imagen para no mostrar hasta el fondo infinito, solo hasta 3 metros
idx_max = np.argmax(profundidades > 3.0)

# Graficamos
plt.imshow(b_scan_fmcw[:idx_max, :],
           extent=[x_antena[0], x_antena[-1], profundidades[idx_max], profundidades[0]],
           aspect='auto', cmap='magma')

plt.title('Radargrama FMCW (Construido mediante Transformadas de Fourier)', fontsize=14)
plt.xlabel('Posición de la antena en el techo (m)', fontsize=12)
plt.ylabel('Profundidad en la roca (m)', fontsize=12)
plt.colorbar(label='Amplitud del Eco (Magnitud FFT)')

# Marcamos la posición real del hueco
plt.plot(x_objetivo, z_objetivo, 'w+', markersize=12, label='Posición Real del Hueco')
plt.legend()
plt.grid(color='white', linestyle='--', alpha=0.3)
plt.show()