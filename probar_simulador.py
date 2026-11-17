import sys
import time

import trimesh

from core.config import ConfigEscena
from core.simulator import simular_escena, verificar_escena

ruta = sys.argv[1]
malla = trimesh.load(ruta, force='mesh')

config = ConfigEscena()
t0 = time.time()
r = simular_escena(malla.vertices, malla.faces, config, progreso=lambda f, texto: print(f'{f:4.0%}  {texto}'))
g = r.geometria

print(f'Tiempo: {time.time() - t0:.1f} s')
print(f'Ω_cam = {g.omega_cam.sum():,} px | Ω_map = {g.omega_map.sum():,} px ({100 * g.omega_map.sum() / g.omega_cam.sum():.1f} % de Ω_cam)')
print(f'Saturados: {r.mascara_saturacion.sum():,} px | r_ideal = {r.r_ideal:.4f}')
print('Problemas:', verificar_escena(r) or 'ninguno')