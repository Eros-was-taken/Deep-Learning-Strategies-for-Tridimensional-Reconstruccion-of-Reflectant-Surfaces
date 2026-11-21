import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from core.simulator import verificar_escena

VERSION_FORMATO = 1

def nombre_escena(config):
    o = config.objeto
    return f'escena_a{o.alpha_deg:g}_b{o.beta_deg:g}'

def _guardar_png16(imagen, ruta):
    datos = np.rint(np.clip(imagen, 0.0, 1.0) * 65535.0).astype(np.uint16)
    Image.fromarray(datos).save(ruta)

def _guardar_png8(arreglo, ruta):
    Image.fromarray(np.ascontiguousarray(arreglo, dtype=np.uint8)).save(ruta)

def exportar_escena(resultado, carpeta_padre, sobrescribir=False, progreso=None):
    cfg, g = resultado.config, resultado.geometria
    carpeta = Path(carpeta_padre) / nombre_escena(cfg)
    if carpeta.exists() and any(carpeta.iterdir()) and not sobrescribir:
        raise FileExistsError(f'La carpeta ya existe y no está vacía: {carpeta}')

    dir_ideales = carpeta / 'imagenes_ideales'
    dir_finales = carpeta / 'imagenes_finales'
    dir_mascaras = carpeta / 'mascaras'
    for d in (dir_ideales, dir_finales, dir_mascaras):
        d.mkdir(parents=True, exist_ok=True)
    
    n = len(resultado.imagenes_finales)
    for k in range(n):
        if progreso is not None:
            progreso(k / n, f'Guardando imagen {k + 1} de {n}')
        _guardar_png16(resultado.imagenes_ideales[k], dir_ideales / f'ideal_{k + 1:02d}.png')
        _guardar_png16(resultado.imagenes_finales[k], dir_finales / f'franja_{k + 1:02d}.png')

    _guardar_png8(g.omega_cam * 255, dir_mascaras / 'omega_cam.png')
    _guardar_png8(g.omega_map * 255, dir_mascaras / 'omega_map.png')
    _guardar_png8(resultado.mascara_saturacion * 255, dir_mascaras / 'saturacion.png')
    _guardar_png8(g.etiquetas, dir_mascaras / 'etiquetas.png')

    np.savez_compressed(carpeta / 'referencias.npz', fase_continua=resultado.fase_continua.astype(np.float32), fase_ideal=resultado.fase_ideal.astype(np.float32),
                        puntos_3d=g.puntos_3d.astype(np.float32), normales=g.normales.astype(np.float32), u_p=g.u_p.astype(np.float32), v_p=g.v_p.astype(np.float32),)
    
    cfg.guardar_json(carpeta / 'config.json')

    problemas = verificar_escena(resultado)
    metadatos = {
        'version_formato': VERSION_FORMATO,
        'fecha': datetime.now().isoformat(timespec='seconds'),
        'n_imagenes': n,
        'dimensiones': {'alto': int(g.omega_cam.shape[0]), 'ancho': int(g.omega_cam.shape[1])},
        'pixeles': {
            'omega_cam': int(g.omega_cam.sum()),
            'omega_map': int(g.omega_map.sum()),
            'saturados': int(resultado.mascara_saturacion.sum()),
        },
        'r_ideal': float(resultado.r_ideal),
        'problemas': problemas,
        'convenciones': {
            'png16': 'intensidad = valor / 65535, en [0, 1]',
            'mascaras': '0 = no, 255 = si',
            'etiquetas': '0 fuera de Ω_cam, 1 iluminado (Ω_map), 2 fuera del campo del proyector, 3 sombra propia, 4 sombra proyectada',
            'referencias.npz': 'float32; fases en rad con NaN fuera de Ω_map; puntos_3d y profundidad en mm, marco de la cámara; u_p y v_p con origen en 1'
        },
    }
    with open(carpeta / 'metadatos.json', 'w', encoding='utf-8') as fh:
        json.dump(metadatos, fh, indent=2, ensure_ascii=False)
    if progreso is not None:
        progreso(1.0, 'Listo')
    return carpeta, metadatos

def cargar_secuencia(carpeta, subcarpeta='imagenes_finales'):
    archivos = sorted((Path(carpeta) / subcarpeta).glob('*.png'))
    if not archivos:
        raise FileNotFoundError(f'No hay imágenes en {Path(carpeta) / subcarpeta}')
    pila = [np.asarray(Image.open(a), dtype=np.float32) / 65535.0 for a in archivos]
    return np.stack(pila)