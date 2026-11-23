import sys
import tempfile
import traceback

import trimesh

from core.config import ConfigEscena
from core.exportacion import exportar_niveles
from core.simulator import iterar_niveles

malla = trimesh.load(sys.argv[1], force='mesh')
config = ConfigEscena()
niveles = ['low', 'moderate']

try:
    for nombre, r in iterar_niveles(malla.vertices, malla.faces, config, niveles):
        print(nombre, 'saturados:', int(r.mascara_saturacion.sum()))
    carpeta, meta = exportar_niveles(
        iterar_niveles(malla.vertices, malla.faces, config, niveles),
        tempfile.mkdtemp(), 'prueba', config_base=config)
    print('Exportación OK:', carpeta)
except Exception:
    traceback.print_exc()