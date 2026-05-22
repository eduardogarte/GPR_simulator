import numpy as np
import matplotlib.pyplot as plt

# 1. Parámetros del entorno
c = 0.3  # Velocidad de la luz en m/ns (aprox)
epsilon_suelo = 6.0  # Suelo seco
miu = 1 # H/m free space magmetic permeability
epsilon_mina = 3.0   # Plástico/TNT
profundidad_mina = 5 # metros
Z0 = 377 # free space impedance

# -- WAve properies in perspective
#v = [0.0, 15] meters/ns range or v = [0.2, 0.5] when is normalized to the velocity in the air
#attenuation = 1dBm with high loss of 10-100 dB and very low loss setting being 0.01–0.1 dB/m
# impedance = [100, 150] ohms
# 2. Calcular velocidades y coeficientes
sigma = 0
v_suelo = c / np.sqrt(miu*epsilon_suelo) # GPR book, 1.14/1.16 equation
attenuation = Z0 * sigma/ 2 *(np.sqrt(miu*epsilon_suelo)) #1.18 equation
impedance = Z0/(2*(np.sqrt(miu*epsilon_suelo))) # 1.9 equation
tiempo_ida_vuelta = (2 * profundidad_mina) / v_suelo
coef_reflexion = (np.sqrt(epsilon_suelo) - np.sqrt(epsilon_mina)) / (np.sqrt(epsilon_suelo) + np.sqrt(epsilon_mina))

# 3. Crear el pulso (Ricker Wavelet - El estándar en GPR)
def ricker_wavelet(f, t):
    tau = np.pi * f * t
    return (1 - 2 * tau**2) * np.exp(-tau**2)

f_central = 1.0  # 1 GHz (común para minas)
t = np.linspace(0, 100, 1000) # nanosegundos

# 4. Generar la señal recibida (A-scan)
# Pulso inicial (superficie) + Pulso reflejado (mina) con atenuación simple
pulso_inicial = ricker_wavelet(f_central, t - 1)
eco_mina = coef_reflexion * ricker_wavelet(f_central, t - 1 - tiempo_ida_vuelta) * 0.7

traza = pulso_inicial + eco_mina

# Visualización
plt.figure(figsize=(10, 5))
plt.subplot(221)
plt.plot(t, traza, label="Señal GPR (A-scan)")
plt.axvline(tiempo_ida_vuelta + 1, color='r', linestyle='--', label="Eco de la Mina")
plt.title("Simulación de detección de mina (Pulso reflejado)")
plt.xlabel("Tiempo (ns)")
plt.ylabel("Echo Amplitud (D.C.)")
plt.legend()
plt.grid(True)


# Parámetros físicos
epsilon_r = 6.0 # suelo seco
v = c / np.sqrt(epsilon_r)

# Configuración del escenario
x_antena = np.linspace(0, 2, 100)  # El radar se mueve de 0 a 2 metros
tiempo = np.linspace(0, 20, 400)  # Ventana de tiempo de 20 ns
pos_mina = 1.0  # Mina en x = 1m
prof_mina = 0.6  # Mina a 60cm de profundidad

# Crear matriz para el B-scan (Tiempo x Posición)
b_scan = np.zeros((len(tiempo), len(x_antena)))

# Simulación: Generar un A-scan para cada posición de la antena
for i, x in enumerate(x_antena):
    # Calcular tiempo de llegada desde la mina
    distancia = np.sqrt((x - pos_mina) ** 2 + prof_mina ** 2)
    t_eco = (2 * distancia) / v

    # Añadir el pulso reflejado a la matriz
    b_scan[:, i] = ricker_wavelet(f_central, tiempo - t_eco)

# Visualización
plt.subplot(222)
plt.imshow(b_scan, extent=[0, 2, 20, 0], aspect='auto', cmap='Greys')
plt.title("Simulación B-scan de una Mina Terrestre")
plt.xlabel("Posición de la antena (m)")
plt.ylabel("Tiempo de viaje (ns)")
plt.colorbar(label="Amplitud")
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE CAPAS ---
# Puedes agregar o quitar elementos de esta lista
suelo = [
    {"nombre": "Aire", "espesor": 0.2, "eps": 1.0},  # Capa de aire (antena elevada)
    {"nombre": "Tierra Seca", "espesor": 0.4, "eps": 4.0},
    {"nombre": "Tierra Húmeda", "espesor": 0.5, "eps": 12.0},
]


def calcular_tiempos_y_reflexiones(capas, c=0.3):
    tiempos_interfaces = []
    coeficientes_R = []
    tiempo_acumulado = 0

    for i in range(len(capas) - 1):
        eps_actual = capas[i]["eps"]
        eps_siguiente = capas[i + 1]["eps"]
        espesor = capas[i]["espesor"]

        # 1. Velocidad en la capa actual (m/ns)
        v = c / np.sqrt(eps_actual)

        # 2. Tiempo de ida y vuelta en esta capa
        tiempo_acumulado += (2 * espesor) / v
        tiempos_interfaces.append(tiempo_acumulado)

        # 3. Coeficiente de reflexión en la frontera (Ecuación de Fresnel)
        R = (np.sqrt(eps_actual) - np.sqrt(eps_siguiente)) / (np.sqrt(eps_actual) + np.sqrt(eps_siguiente))
        coeficientes_R.append(R)

    return tiempos_interfaces, coeficientes_R


# --- SIMULACIÓN DE A-SCAN ---
t = np.linspace(0, 30, 1500)
f_central = 1.0


def generar_señal(tiempos, reflectividades, t_vector):
    señal = np.zeros_like(t_vector)
    # Pulso inicial (superficie)
    señal += ricker_wavelet(f_central, t_vector - 1)

    # Ecos de cada interfaz
    for tiempo, R in zip(tiempos, reflectividades):
        señal += R * ricker_wavelet(f_central, t_vector - 1 - tiempo)

    return señal


# Ejecución
tiempos, Rs = calcular_tiempos_y_reflexiones(suelo)
traza = generar_señal(tiempos, Rs, t)

# Visualización
plt.figure(figsize=(10, 4))
plt.plot(t, traza,label="Señal GPR (A-scan)")
plt.axvline(tiempos[0]+1, color='r', linestyle='--', label="Eco de la Mina")
plt.axvline(tiempos[1]+1, color='r', linestyle='--', label="Primer eco de la Mina")
plt.title("A-scan con Múltiples Capas Geológicas")
plt.xlabel("Tiempo (ns)")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
c = 0.3  # Velocidad luz m/ns
f_central = 1.2  # 1.2 GHz (alta resolución para minas)

# Definimos las capas: [permitividad, espesor]
# Capa 0: Aire (eps=1), Capa 1: Tierra seca (eps=4)
eps_capas = [1.0, 4.0, 12.0]
espesores = [0.2, 0.4, 0.5]  # m

# Posición de la mina (en el medio del recorrido y a 0.3m de profundidad total)
x_mina = 1.0
z_mina = 0.3  # Está dentro de la segunda capa (Tierra seca)

# Parámetros del escaneo
x_pasos = np.linspace(0, 2, 100)  # 2 metros de recorrido
tiempo = np.linspace(0, 15, 400)  # 15 nanosegundos de registro
b_scan = np.zeros((len(tiempo), len(x_pasos)))



# --- SIMULACIÓN ---
for i, x_ant in enumerate(x_pasos):
    traza = np.zeros_like(tiempo)

    # 1. Ecos de las Capas (Horizontales)
    # Interfaz 1 (Aire-Suelo)
    v0 = c / np.sqrt(eps_capas[0])
    t_capa1 = (2 * espesores[0]) / v0
    v_capa2 = c / np.sqrt(eps_capas[2])
    t_capa2 =  (2 * espesores[1]) / v_capa2
    R1 = (1 - 2) / (1 + 2)  # Simplificado
    traza += R1 * ricker_wavelet(f_central, tiempo - 1 - t_capa1)
    traza += R1 * ricker_wavelet(f_central, tiempo - 1 - t_capa2)


    # 2. Eco de la Mina (Hipérbola)
    # Calculamos la velocidad promedio hasta la mina para simplificar
    v_promedio = c / np.sqrt(eps_capas[1])
    distancia = np.sqrt((x_ant - x_mina) ** 2 + z_mina ** 2)
    t_mina = (2 * distancia) / v_promedio



    # Simulamos el eco de la mina (más débil que el suelo)
    traza += 0.2 * ricker_wavelet(f_central, tiempo - 1 - t_mina)

    b_scan[:, i] = traza

# --- VISUALIZACIÓN ---
plt.figure(figsize=(10, 6))
# Usamos un filtro para resaltar los cambios (Background removal simple)
plt.imshow(b_scan, extent=[0, 2, 15, 0], aspect='auto', cmap='RdBu', vmin=-0.5, vmax=0.5)
plt.title("B-scan: Mina enterrada bajo una capa de suelo")
plt.xlabel("Posición (m)")
plt.ylabel("Tiempo (ns)")
plt.colorbar(label="Amplitud de señal")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# --- 1. PARÁMETROS FÍSICOS Y DEL ENTORNO ---
c = 0.3  # Velocidad de la luz en el vacío (m/ns)
f_central = 1.0  # Frecuencia central de la antena (1 GHz)

# Propiedades del Suelo (Aquí controlamos la HUMEDAD)
epsilon_r = 9.0  # Permitividad (Suelo arenoso algo húmedo)
v_suelo = c / np.sqrt(epsilon_r)  # Velocidad en el suelo (aprox 0.1 m/ns)

# Factor de Atenuación por Humedad (Conductividad)
# Un valor bajo (ej. 0.1) es suelo seco. Un valor alto (ej. 0.8) es suelo muy húmedo/salino.
# Esto hará que las señales profundas se debiliten exponencialmente.
FACTOR_HUMEDAD = 0.5

# --- 2. DEFINICIÓN DE OBJETIVOS (LAS "QUEMAS" O MINAS) ---
# Lista de diccionarios con posición X, profundidad Z, y fuerza relativa del rebote
minas = [
    {"x": 0.5, "z": 0.3, "fuerza": 0.4},  # Mina superficial izquierda
    {"x": 1.2, "z": 0.7, "fuerza": 0.4},  # Mina profunda central (se verá más débil por humedad)
    {"x": 1.8, "z": 0.4, "fuerza": 0.4}  # Mina superficial derecha
]

# --- 3. CONFIGURACIÓN DEL ESCANEO ---
x_antena = np.linspace(0, 2.5, 150)  # Recorrido de 2.5 metros
tiempo = np.linspace(0, 25, 500)  # Ventana de tiempo de 25 ns
dt = tiempo[1] - tiempo[0]

# Matriz para el Radargrama Crudo
b_scan_raw = np.zeros((len(tiempo), len(x_antena)))


# Función del pulso (Ricker Wavelet)
def ricker(t_vector, t_llegada, f):
    t_offset = t_vector - t_llegada - 1.0  # El -1.0 es un pequeño retraso inicial
    tau = np.pi * f * t_offset
    return (1 - 2 * tau ** 2) * np.exp(-tau ** 2)


# --- 4. SIMULACIÓN PRINCIPAL (Generar el Radargrama Crudo) ---
print("Simulando escaneo GPR...")
for i, x_pos in enumerate(x_antena):
    traza = np.zeros_like(tiempo)

    # A) Agregar el "Rebote del Suelo" (Surface Clutter)
    # Es una señal fuerte y plana siempre al mismo tiempo inicial
    t_superficie = 1.5  # ns
    traza += 1.0 * ricker(tiempo, t_superficie, f_central)

    # B) Agregar los ecos de las MINAS
    for mina in minas:
        # Distancia de ida y vuelta (teorema de Pitágoras)
        distancia_total = 2 * np.sqrt((x_pos - mina["x"]) ** 2 + mina["z"] ** 2)
        t_vuelo = distancia_total / v_suelo + t_superficie

        # --- APLICACIÓN DE LA HUMEDAD (Atenuación Exponencial) ---
        # La señal decae según la distancia recorrida y el factor de humedad
        atenuacion = np.exp(-FACTOR_HUMEDAD * distancia_total)

        amplitud_final = mina["fuerza"] * atenuacion

        traza += amplitud_final * ricker(tiempo, t_vuelo, f_central)

    b_scan_raw[:, i] = traza

# --- 5. PROCESAMIENTO DE SEÑAL: FILTRO "BACKGROUND REMOVAL" ---
print("Aplicando filtro de eliminación de fondo...")

# Paso 1: Calcular la traza promedio de todo el escaneo
traza_promedio = np.mean(b_scan_raw, axis=1)

# Paso 2: Restar el promedio a cada traza individual
# (Usamos np.newaxis para poder restar un vector columna a la matriz)
b_scan_filtrado = b_scan_raw - traza_promedio[:, np.newaxis]

# --- 6. VISUALIZACIÓN (RADARGRAMAS) ---
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Límites para los ejes de la gráfica
extent = [x_antena[0], x_antena[-1], tiempo[-1], tiempo[0]]

# Plot 1: Radargrama Crudo
# Usamos 'seismic' o 'RdBu' que son mapas de colores comunes en geofísica
im1 = axes[0].imshow(b_scan_raw, aspect='auto', extent=extent, cmap='seismic', vmin=-0.8, vmax=0.8)
axes[0].set_title(f"Radargrama Crudo (Suelo con Factor Humedad={FACTOR_HUMEDAD})")
axes[0].set_ylabel("Tiempo de viaje (ns) / ~Profundidad")
axes[0].set_xlabel("Posición de la antena (metros)")
fig.colorbar(im1, ax=axes[0], label="Amplitud")

# Plot 2: Radargrama Filtrado
# Notar que usamos un vmin/vmax menor para aumentar el contraste de las hipérbolas débiles
im2 = axes[1].imshow(b_scan_filtrado, aspect='auto', extent=extent, cmap='seismic', vmin=-0.2, vmax=0.2)
axes[1].set_title("Radargrama Filtrado (Background Removal Aplicado)")
axes[1].set_ylabel("Tiempo de viaje (ns)")
axes[1].set_xlabel("Posición de la antena (metros)")
fig.colorbar(im2, ax=axes[1], label="Amplitud (Contrastada)")

plt.tight_layout()
plt.show()