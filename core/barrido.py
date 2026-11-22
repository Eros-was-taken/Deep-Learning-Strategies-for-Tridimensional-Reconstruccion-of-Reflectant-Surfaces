import copy
import json
from datetime import datetime
from pathlib import Path

from core.config import ConfigEscena
from core.exportacion import exportar_escena, nombre_escena
from core.simulator import simular_escena

def generar_orientaciones(n_alpha=8, paso_alpha=45.0, n_beta=24, paso_beta=15.0, alpha0=0.0, beta0=0.0):
    return [(round(alpha0 + i * paso_alpha, 6), round(beta0 + j * paso_beta, 6)) for i in range(n_alpha) for j in range(n_beta)]

def _misma_config(carpeta, config):
    try:
        return ConfigEscena.cargar_json(carpeta / 'config.json') == config
    except (OSError, ValueError, TypeError):
        return False

def barrer_orientaciones(vertices, faces, config_base, orientaciones, carpeta_salida, nombre_modelo='modelo', reanudar=True, progreso=None, cancelar=None):
    destino = Path(carpeta_salida) / nombre_modelo
    destino.mkdir(parents=True, exist_ok=True)
    total = len(orientaciones)
    resumen = {'modelo': nombre_modelo, 'total': total, 'exportadas': [], 'omitidas': [], 'fallidas': [], 'con_problemas': [], 'cancelado': False}
    
    hechas = 0
    for alpha, beta in orientaciones:
        if cancelar is not None and cancelar():
            resumen['cancelado'] = True
            break
        cfg = copy.deepcopy(config_base)
        cfg.objeto.alpha_deg = float(alpha)
        cfg.objeto.beta_deg = float(beta)
        nombre = nombre_escena(cfg)
        if progreso is not None:
            progreso(hechas, total, f'Escenas {hechas + 1} de {total}: α = {alpha:g}°, β = {beta:g}°')
        
        carpeta = destino / nombre
        if reanudar and (carpeta / 'metadatos.json').exists():
            if _misma_config(carpeta, cfg):
                resumen['omitidas'].append(nombre)
            else:
                resumen['fallidas'].append({'escena': nombre, 'error': 'ya existe con otra configuración; usa otra carpeta o desactiva "reanudar" para sobrescribir'})
            hechas += 1
            continue

        try:
            resultado = simular_escena(vertices, faces, cfg)
            _, meta = exportar_escena(resultado, destino, sobrescribir=True)
        except Exception as error:
            resumen['fallidas'].append({'escena': nombre, 'error': f'{type(error).__name__}: {error}'})
        else:
            resumen['exportadas'].append(nombre)
            if meta['problemas']:
                resumen['con_problemas'].append({'escena': nombre, 'problemas': meta['problemas']})
        hechas += 1
    
    if progreso is not None:
        progreso(hechas, total, 'Cancelado' if resumen['cancelado'] else 'Terminado')
    
    indice = {'fecha': datetime.now().isoformat(timespec='seconds'), 'config_base': config_base.a_diccionario(), **resumen}
    with open(destino / 'barrido.json', 'w', encoding='utf-8') as fh:
        json.dump(indice, fh, indent=2, ensure_ascii=False)
    return resumen