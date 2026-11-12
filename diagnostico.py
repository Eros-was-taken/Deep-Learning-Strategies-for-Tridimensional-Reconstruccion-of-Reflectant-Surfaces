import sys
import numpy as np

from core.mesh_loader import load_mesh
from core.camera import build_intrinsic_matrix, build_projector_extrinsic
from core.geometry import condition_mesh
from core.visibility import (rasterize_zbuffer, compute_point_map_and_normals, camera_to_projector_coords)

ruta = sys.argv[1]
mesh, _ = load_mesh(ruta)

# Mismos valores por defecto de la aplicacion
Dc, Dp, ang = 1000.0, 1000.0, 12.0
K, f_px, cx, cy = build_intrinsic_matrix(12.0, 4.65, 1024, 768)
Kp, fp_px, cxp, cyp = build_intrinsic_matrix(12.0, 4.50, 1024, 768)
_, R_po, t_po = build_projector_extrinsic(Dp, ang)

print("f_px camara =", round(f_px, 2), "| f_px proyector =", round(fp_px, 2))
print("Kp =\n", Kp)
print("R_po forma:", np.shape(R_po), "| t_po forma:", np.shape(t_po))

V = np.array(mesh.vertices)
vcam, scale, Lx, Ly = condition_mesh(V, 0.0, 0.0, f_px, 1024, 768, Dc)
print("\nVertices en camara: min", vcam.min(axis=0).round(1), "max", vcam.max(axis=0).round(1))
print("Extension (mm):", (vcam.max(axis=0) - vcam.min(axis=0)).round(1))

# Camino 1: directamente con los vertices (sin z-buffer)
u1, v1, z1 = camera_to_projector_coords(vcam, R_po, t_po, Kp, Dc)
print("\n[1] Desde vertices: u_p en [%.0f, %.0f], v_p en [%.0f, %.0f], z_p en [%.0f, %.0f]"
      % (u1.min(), u1.max(), v1.min(), v1.max(), z1.min(), z1.max()))

# Centro del objeto -> debe caer en el punto principal del proyector (512.5, 384.5)
centro = np.array([[0.0, 0.0, Dc]])
uc, vc, zc = camera_to_projector_coords(centro, R_po, t_po, Kp, Dc)
print("    Centro del objeto -> (%.1f, %.1f), z_p=%.1f (esperado 512.5, 384.5, 1000)" % (uc[0], vc[0], zc[0]))

# Camino 2: con el mapa de puntos (z-buffer)
faces = np.array(mesh.faces)
depth, fid, om = rasterize_zbuffer(vcam, faces, K, 1024, 768)
P, N = compute_point_map_and_normals(vcam, faces, fid, depth, K)
print("\n[2] Mapa de puntos: X en [%.1f, %.1f], Y en [%.1f, %.1f], Z en [%.1f, %.1f]"
      % (np.nanmin(P[..., 0]), np.nanmax(P[..., 0]), np.nanmin(P[..., 1]),
         np.nanmax(P[..., 1]), np.nanmin(P[..., 2]), np.nanmax(P[..., 2])))
u2, v2, z2 = camera_to_projector_coords(P, R_po, t_po, Kp, Dc)
dentro = om & (u2 >= 1) & (u2 <= 1024) & (v2 >= 1) & (v2 <= 768)
print("    Desde mapa: u_p en [%.0f, %.0f]; dentro del proyector: %d de %d (%.1f %%)"
      % (np.nanmin(u2), np.nanmax(u2), dentro.sum(), om.sum(), 100 * dentro.sum() / om.sum()))