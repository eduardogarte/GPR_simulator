import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINICIÓN DEL ESPACIO 3D
# ==========================================
# Creamos una cuadrícula de 5x5 metros en superficie y 3 metros de profundidad
nx, ny, nz = 60, 60, 50
x = np.linspace(0, 5, nx)
y = np.linspace(0, 5, ny)
z = np.linspace(0, 3, nz)

# Nuestro "Cubo de Datos" GPR (Matriz 3D) [Profundidad, Y, X]
cubo_gpr = np.zeros((nz, ny, nx))

# ==========================================
# 2. CREACIÓN DE LA GEOMETRÍA DE LOS ECOS 3D
# ==========================================
print("Generando volumen GPR 3D...")
# Objeto 1: Una tubería cruzando en diagonal a 1.2 m de profundidad
# Objeto 2: Una roca esférica (hueco) en x=4, y=1, z=2.0 m

for i in range(nx):
    for j in range(ny):
        # Posición de la antena en superficie
        ax, ay = x[i], y[j]

        # --- Eco de la Tubería Diagonal ---
        # La ecuación de una línea diagonal simple en el suelo: y = x
        # Distancia mínima de la antena (ax, ay) a la línea y=x en el plano 2D
        dist_plano_tuberia = np.abs(ax - ay) / np.sqrt(2)
        profundidad_tuberia = 1.2

        # Distancia espacial 3D (Hipotenusa)
        dist_3d_tuberia = np.sqrt(dist_plano_tuberia ** 2 + profundidad_tuberia ** 2)

        # --- Eco de la Roca ---
        rx, ry, rz = 4.0, 1.0, 2.0
        dist_3d_roca = np.sqrt((ax - rx) ** 2 + (ay - ry) ** 2 + rz ** 2)

        # "Pintamos" los ecos en el índice de profundidad Z correspondiente
        # Convertimos la distancia 3D a un índice en el eje Z de nuestra matriz
        idx_z_tuberia = int((dist_3d_tuberia / 3.0) * nz)
        idx_z_roca = int((dist_3d_roca / 3.0) * nz)

        if idx_z_tuberia < nz:
            cubo_gpr[idx_z_tuberia, j, i] += 1.0  # Tubería fuerte
        if idx_z_roca < nz:
            cubo_gpr[idx_z_roca, j, i] += 0.8  # Roca un poco más débil

# ==========================================
# 3. EXTRACCIÓN DE LAS REBANADAS (SLICING)
# ==========================================
# A) Rebanada Frontal (B-Scan) fijando Y = 2.5 metros (Mitad del terreno)
indice_y_fijo = ny // 2
b_scan_frontal = cubo_gpr[:, indice_y_fijo, :]

# B) Rebanada de Profundidad (C-Scan) fijando Z = 1.2 metros (Donde está la tubería)
indice_z_tubo = int((1.2 / 3.0) * nz)
c_scan_profundidad = cubo_gpr[indice_z_tubo, :, :]

# C) Rebanada de Profundidad (C-Scan) fijando Z = 2.0 metros (Donde está la roca)
indice_z_roca = int((2.0 / 3.0) * nz)
c_scan_roca = cubo_gpr[indice_z_roca, :, :]

# ==========================================
# 4. VISUALIZACIÓN MULTIPANEL
# ==========================================
fig = plt.figure(figsize=(15, 10))

# --- Panel 1: Vista Frontal (B-Scan Clásico) ---
ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(b_scan_frontal, extent=[0, 5, 3, 0], aspect='auto', cmap='magma')
ax1.set_title("Rebanada Frontal (B-Scan) en Y = 2.5m\n(Corte vertical del suelo)")
ax1.set_xlabel("Eje X (m)")
ax1.set_ylabel("Profundidad Z (m)")
# Vemos la hipérbola de la tubería cruzando por este corte

# --- Panel 2: Vista de Profundidad (C-Scan a 1.2m) ---
ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(c_scan_profundidad, extent=[0, 5, 5, 0], aspect='auto', cmap='viridis')
ax2.set_title("Rebanada de Profundidad (C-Scan) a Z = 1.2m\n(Vista desde arriba)")
ax2.set_xlabel("Eje X (m)")
ax2.set_ylabel("Eje Y (m)")
# Vemos la línea recta de la tubería brillando

# --- Panel 3: Vista de Profundidad (C-Scan a 2.0m) ---
ax3 = fig.add_subplot(2, 2, 3)
ax3.imshow(c_scan_roca, extent=[0, 5, 5, 0], aspect='auto', cmap='viridis')
ax3.set_title("Rebanada de Profundidad (C-Scan) a Z = 2.0m\n(Vista desde arriba más profundo)")
ax3.set_xlabel("Eje X (m)")
ax3.set_ylabel("Eje Y (m)")
# Vemos el punto brillante de la roca, y ecos fantasma de la tubería expandiéndose

# --- Panel 4: Renderizado 3D de las rebanadas cruzadas ---
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
X_mesh, Y_mesh = np.meshgrid(x, y)
# Dibujamos el C-Scan de la tubería flotando en su profundidad real
ax4.contourf(X_mesh, Y_mesh, c_scan_profundidad, zdir='z', offset=-1.2, cmap='viridis', alpha=0.8)
ax4.set_zlim(-3, 0)
ax4.set_title("Perspectiva 3D del C-Scan en el espacio")
ax4.set_xlabel('X (m)')
ax4.set_ylabel('Y (m)')
ax4.set_zlabel('Profundidad (m)')

plt.tight_layout()
plt.show()