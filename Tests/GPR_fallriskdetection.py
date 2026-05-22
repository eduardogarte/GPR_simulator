import numpy as np
import matplotlib.pyplot as plt

# --- PARÁMETROS FÍSICOS (Basados en literatura minera / NIOSH) ---
c = 0.3  # Velocidad de la luz en el vacío (m/ns)
f_central = 0.9  # 900 MHz (0.9 GHz) - Estándar para inspección de techos mineros
eps_roca = 6.0  # Permitividad relativa de una roca sedimentaria (ej. lutita o arenisca)
eps_aire = 1.0  # Permitividad del aire dentro de la fractura

v_roca = c / np.sqrt(eps_roca)  # Velocidad en la roca ~0.122 m/ns
v_aire = c / np.sqrt(eps_aire)  # Velocidad en el aire = 0.3 m/ns

# Coeficientes de reflexión (Ecuaciones de Fresnel para incidencia normal)
R_roca_aire = (np.sqrt(eps_roca) - np.sqrt(eps_aire)) / (np.sqrt(eps_roca) + np.sqrt(eps_aire))
R_aire_roca = (np.sqrt(eps_aire) - np.sqrt(eps_roca)) / (np.sqrt(eps_aire) + np.sqrt(eps_roca))

# --- MODELO DEL TECHO DE LA MINA ---
x = np.linspace(0, 5, 250)  # Escaneo de 5 metros a lo largo del techo de la galería
tiempo = np.linspace(0, 30, 600)  # 30 ns de ventana temporal
b_scan = np.zeros((len(tiempo), len(x)))

# Geometría de la fractura oculta (Delaminación o desprendimiento)
profundidad_fractura = 1.2  # metros (A esta profundidad, un colapso es fatal)
espesor_fractura = 0.02  # 2 centímetros de aire (Causa desconexión mecánica total)
inicio_fractura_x = 1.5  # Inicia al metro 1.5
fin_fractura_x = 3.5  # Termina al metro 3.5

# Cálculos de tiempos de vuelo analíticos
t_techo_fractura = (2 * profundidad_fractura) / v_roca
dt_gap = (2 * espesor_fractura) / v_aire  # Tiempo extra para cruzar los 2 cm de aire


def ricker(t, t0, f):
    tau = np.pi * f * (t - t0)
    return (1 - 2 * tau ** 2) * np.exp(-tau ** 2)


# Simulación de la Adquisición GPR
np.random.seed(42)  # Ruido natural de fondo
for i, pos_x in enumerate(x):
    traza = np.zeros_like(tiempo)

    # Ruido natural de la roca
    traza += np.random.normal(0, 0.015, len(tiempo))

    # Acople inicial de la antena contra el techo
    traza += 0.8 * ricker(tiempo, 2.0, f_central)

    if inicio_fractura_x <= pos_x <= fin_fractura_x:
        # Atenuación geométrica por propagación en la roca
        atenuacion = np.exp(-0.3 * profundidad_fractura)

        # A) Eco del techo de la fractura (Roca -> Aire)
        eco_superior = R_roca_aire * atenuacion * ricker(tiempo, t_techo_fractura + 2.0, f_central)

        # B) Eco del suelo de la fractura (Aire -> Roca) -> Reflectividad Negativa
        eco_inferior = R_aire_roca * atenuacion * ricker(tiempo, t_techo_fractura + dt_gap + 2.0, f_central)

        traza += eco_superior + eco_inferior

    else:
        # 3. Difracción en los bordes rotos
        dist_inicio = np.sqrt((pos_x - inicio_fractura_x) ** 2 + profundidad_fractura ** 2)
        dist_fin = np.sqrt((pos_x - fin_fractura_x) ** 2 + profundidad_fractura ** 2)

        if pos_x < inicio_fractura_x:
            traza += 0.15 * R_roca_aire * ricker(tiempo, (2 * dist_inicio) / v_roca + 2.0, f_central)
        if pos_x > fin_fractura_x:
            traza += 0.15 * R_roca_aire * ricker(tiempo, (2 * dist_fin) / v_roca + 2.0, f_central)

    b_scan[:, i] = traza


plt.figure(figsize=(10, 6))
# Usamos un filtro para resaltar los cambios (Background removal simple)
plt.imshow(b_scan, extent=[0, 2, 15, 0], aspect='auto', cmap='RdBu', vmin=-0.5, vmax=0.5)
plt.title("B-scan: Mina enterrada bajo una capa de suelo")
plt.xlabel("Posición (m)")
plt.ylabel("Tiempo (ns)")
plt.colorbar(label="Amplitud de señal")
plt.show()