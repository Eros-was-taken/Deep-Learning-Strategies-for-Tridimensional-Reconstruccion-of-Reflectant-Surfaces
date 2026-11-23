import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from core.simulator import nombre_nivel, verificar_escena

VERSION_FORMATO = 2

def nombre_escena(config):
    o = config.objeto
    return f'escena_a{o.alpha_deg:g}_b{o.beta_deg:g}'

def _guardar_png16(imagen, ruta):
    datos = np.rint(np.clip(imagen, 0.0, 1.0) * 65535.0).astype(np.uint16)
    Image.fromarray(datos).save(ruta)

def _guardar_png8(arreglo, ruta):
    Image.fromarray(np.ascontiguousarray(arreglo, dtype=np.uint8)).save(ruta)

def _escribir_comun(resultado, carpeta):
    g = resultado.geometria
    dir_ideales = carpeta / 'imagenes_ideales'
    dir_mascaras = carpeta / 'mascaras'
    dir_ideales.mkdir(exist_ok=True)
    dir_mascaras.mkdir(exist_ok=True)

    for k, imagen in enumerate(resultado.imagenes_ideales):
        _guardar_png16(imagen, dir_ideales / f'ideal_{k + 1:02d}.png')
    
    _guardar_png8(g.omega_cam * 255, dir_mascaras / 'omega_cam.png')
    _guardar_png8(g.omega_map * 255, dir_mascaras / 'omega_map.png')
    _guardar_png8(g.etiquetas, dir_mascaras / 'etiquetas.png')

    np.savez_compressed(carpeta / 'referencias.npz', fase_continua=resultado.fase_continua.astype(np.float32), fase_ideal=resultado.fase_ideal.astype(np.float32),
                        puntos_3d=g.puntos_3d.astype(np.float32), normales=g.normales.astype(np.float32), u_p=g.u_p.astype(np.float32), v_p=g.v_p.astype(np.float32),)
                    
def _escribir_nivel(resultado, carpeta, nombre):
    destino = carpeta / 'degradadas' / nombre
    destino.mkdir(parents=True, exist_ok=True)
    for k, imagen in enumerate(resultado.imagenes_finales):
        _guardar_png16(imagen, destino / f'franja_{k + 1:02d}.png')
    _guardar_png8(resultado.mascara_saturacion * 255, destino / 'saturacion.png')
    resultado.config.guardar_json(destino / 'config.json')

def exportar_niveles(niveles, carpeta_padre, nombre, config_base=None, sobrescribir=False, progreso=None):
    carpeta = Path(carpeta_padre) / nombre
    if carpeta.exists() and any(carpeta.iterdir()):
        if not sobrescribir:
            raise FileExistsError(f'La carpeta ya existe y no está vacía: {carpeta}')
        if not ((carpeta / 'config.json').exists() or (carpeta / 'metadatos.json').exists()):
            raise FileExistsError(f'La carpeta existe y no parece una escena exportada; no se toca {carpeta}')
        shutil.rmtree(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)

    por_nivel = {}
    problemas = []
    primero = None
    for indice, (nivel, resultado) in enumerate(niveles):
        if primero is None:
            primero = resultado
            (config_base or resultado.config).guardar_json(carpeta / 'config.json')
            _escribir_comun(resultado, carpeta)
        if progreso is not None:
            progreso(indice, f'Guardando nivel {nivel}')
        _escribir_nivel(resultado, carpeta, nivel)
        d = resultado.config.degradacion
        por_nivel[nivel] = {
            'degradacion': {'g': d.g, 'o': d.o, 'sigma_esp_px': d.sigma_esp_px, 'kappa': d.kappa, 'sigma': d.sigma, 'gamma': d.gamma, 'semilla': d.semilla},
            'saturados': int(resultado.mascara_saturacion.sum()),
        }
        for p in verificar_escena(resultado):
            if p not in problemas:
                problemas.append(p)
        
    if primero is None:
        raise ValueError('No se recibió ningún nivel para exportar')

    g = primero.geometria
    metadatos = {
        'version_formato': VERSION_FORMATO,
        'fecha': datetime.now().isoformat(timespec='seconds'),
        'n_imagenes': int(len(primero.imagenes_ideales)),
        'dimensiones': {'alto': int(g.omega_cam.shape[0]), 'ancho': int(g.omega_cam.shape[1])},
        'pixeles': {
            'omega_cam': int(g.omega_cam.sum()),
            'omega_map': int(g.omega_map.sum()),
        },
        'r_ideal': float(primero.r_ideal),
        'niveles': por_nivel,
        'problemas': problemas,
        'convenciones': {
            'png16': 'intensidad = valor / 65535, en [0, 1]',
            'mascaras': '0 = no, 255 = si',
            'etiquetas': '0 fuera de Ω_cam, 1 iluminado (Ω_map), 2 fuera del campo del proyector, 3 sombra propia, 4 sombra proyectada',
            'referencias.npz': 'float32; fases en rad con NaN fuera de Ω_map; puntos_3d y profundidad en mm, marco de la cámara; u_p y v_p con origen en 1',
            'degradadas': 'cada nivel tiene su config.json completo, sus 12 imágenes finales y su máscara de saturación',
        },
    }
    with open(carpeta / 'metadatos.json', 'w', encoding='utf-8') as fh:
        json.dump(metadatos, fh, indent=2, ensure_ascii=False)
    if progreso is not None:
        progreso(len(por_nivel), 'Listo')
    return carpeta, metadatos

def exportar_escena(resultado, carpeta_padre, sobrescribir=False, progreso=None):
    nivel = nombre_nivel(resultado.config.degradacion)
    return exportar_niveles([(nivel, resultado)], carpeta_padre, nombre_escena(resultado.config), sobrescribir=sobrescribir, progreso=progreso)

def cargar_secuencia(carpeta, subcarpeta='imagenes_finales'):
    archivos = sorted((Path(carpeta) / subcarpeta).glob('franja_*.png')) or sorted((Path(carpeta) / subcarpeta).glob('ideal_*.png'))
    if not archivos:
        raise FileNotFoundError(f'No hay imágenes en {Path(carpeta) / subcarpeta}')
    pila = [np.asarray(Image.open(a), dtype=np.float32) / 65535.0 for a in archivos]
    return np.stack(pila)