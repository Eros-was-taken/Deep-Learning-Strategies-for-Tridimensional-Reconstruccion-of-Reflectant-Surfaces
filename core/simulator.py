import copy
from dataclasses import dataclass, field

import numpy as np

from core.camera import build_intrinsic_matrix, build_projector_extrinsic
from core.config import ConfigEscena
from core.geometry import condition_mesh
from core.phase import geometric_phase, phase_coverage, psa_phase
from core.radiometry import (NIVELES_ESPECULARES, degrade_fringe_image, expand_specular, ideal_fringe_image, saturated_pixels, specular_map)
from core.visibility import (ETQ_ILUMINADO, compute_point_map_and_normals, compute_projector_visibility, rasterize_zbuffer)

@dataclass
class Geometria:
    omega_cam: np.ndarray
    omega_map: np.ndarray
    etiquetas: np.ndarray
    profundidad: np.ndarray
    puntos_3d: np.ndarray
    normales: np.ndarray
    u_p: np.ndarray
    v_p: np.ndarray
    n_dot_l: np.ndarray
    centro_proyector: np.ndarray

@dataclass
class ResultadoEscena:
    config: ConfigEscena
    geometria: Geometria
    imagenes_ideales: np.ndarray
    imagenes_finales: np.ndarray
    fase_continua: np.ndarray
    fase_ideal: np.ndarray
    mascara_saturacion: np.ndarray
    r_ideal: float
    etapas: dict = field(default_factory=dict)

def _avisar(progreso, fraccion, texto):
    if progreso is not None:
        progreso(fraccion, texto)

@dataclass
class GeometriaCamara:
    vertices_cam: np.ndarray
    faces: np.ndarray
    depth: np.ndarray
    face_id: np.ndarray
    omega_cam: np.ndarray
    P: np.ndarray
    N: np.ndarray

def calcular_geometria_camara(vertices, faces, config, progreso=None):
    cam, obj = config.camara, config.objeto
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)

    _avisar(progreso, 0.0, 'Acondicionando la malla')
    K, f_px, _, _ = build_intrinsic_matrix(cam.focal_mm, cam.pixel_um, cam.ancho, cam.alto)
    vertices_cam, _, _, _ = condition_mesh(vertices, obj.alpha_deg, obj.beta_deg, f_px, cam.ancho, cam.alto, cam.distancia_mm)

    _avisar(progreso, 0.15, 'Proyectando a la cámara')
    depth, face_id, omega_cam = rasterize_zbuffer(vertices_cam, faces, K, cam.ancho, cam.alto)
    P, N = compute_point_map_and_normals(vertices_cam, faces, face_id, depth, K)
    return GeometriaCamara(vertices_cam, faces, depth, face_id, omega_cam, P, N)

def calcular_visibilidad_proyector(vertices_cam, faces, P, N, omega_cam, config, progreso=None):
    cam, pro = config.camara, config.proyector
    _avisar(progreso, 0.35, 'Visibilidad del proyector y sombras')
    K_p, _, _, _ = build_intrinsic_matrix(pro.focal_mm, pro.pixel_um, pro.ancho, pro.alto)
    _, R_po, t_po = build_projector_extrinsic(pro.distancia_mm,  pro.angulo_deg)
    return compute_projector_visibility(vertices_cam, faces, P, N, omega_cam, R_po, t_po, K_p, cam.distancia_mm, pro.ancho, pro.alto)

def calcular_geometria(vertices, faces, config, progreso=None):
    gc = calcular_geometria_camara(vertices, faces, config, progreso)
    etiquetas, info = calcular_visibilidad_proyector(gc.vertices_cam, gc.faces, gc.P, gc.N, gc.omega_cam, config, progreso)
    return Geometria(omega_cam=gc.omega_cam, omega_map=(etiquetas == ETQ_ILUMINADO), etiquetas=etiquetas, profundidad=gc.depth, puntos_3d=gc.P, normales=gc.N, u_p=info['u_p'],
                    v_p=info['v_p'], n_dot_l=info['n_dot_l'], centro_proyector=np.asarray(info['C_proy'], dtype=float),)

def generar_franjas(geometria, config, guardar_etapas=False, progreso=None):
    fr, dg = config.franjas, config.degradacion
    g = geometria
    n = fr.n_pasos
    alto, ancho = g.omega_cam.shape

    S = specular_map(g.puntos_3d, g.normales, g.centro_proyector, g.omega_map, kappa=dg.kappa)
    S_exp = expand_specular(S, g.omega_map, dg.sigma_esp_px)

    ideales64 = np.empty((n, alto, ancho), dtype=float)
    finales = np.empty((n, alto, ancho), dtype=np.float32)
    saturada = np.zeros((alto, ancho), dtype=bool)
    etapas = {c: np.empty((n, alto, ancho), dtype=np.float32) for c in ('ruido', 'gamma', 'pre')} if guardar_etapas else {}

    for k in range(1, n + 1):
        _avisar(progreso, 0.5 + 0.4 * (k - 1) / n, f'Imagen {k} de {n}')
        ideal, _, _ = ideal_fringe_image(g.u_p, g.n_dot_l, g.omega_cam, g.omega_map, period_px=fr.periodo_px, u0_px=fr.u0_px, k=k, n_steps=n, ambient=fr.ambiente)
        et = degrade_fringe_image(ideal, g.omega_map, S_exp, sigma=dg.sigma, gamma=dg.gamma, seed=dg.semilla, k=k, g=dg.g, o=dg.o)
        ideales64[k - 1] = ideal
        finales[k - 1] = et['final']
        saturada |= saturated_pixels(et['pre'], g.omega_map)
        for clave in etapas:
            etapas[clave][k - 1] = et[clave]
        
    _avisar(progreso, 0.92, 'Fase de referencia')
    fase_cont = geometric_phase(g.u_p, g.omega_map, fr.periodo_px, fr.u0_px)
    fase_ideal = psa_phase(ideales64, g.omega_map)

    _avisar(progreso, 1.0, 'Listo')
    return ResultadoEscena(config=config, geometria=g, imagenes_ideales=ideales64.astype(np.float32), imagenes_finales=finales, fase_continua=fase_cont, fase_ideal=fase_ideal,
                        mascara_saturacion=saturada, r_ideal=phase_coverage(fase_cont, g.omega_map), etapas=etapas,)

def simular_escena(vertices, faces, config, guardar_etapas=False, progreso=None):
    errores = config.validar()
    if errores:
        raise ValueError('Configuración no válida: ' + '; '.join(errores))
    geometria = calcular_geometria(vertices, faces, config, progreso)
    return generar_franjas(geometria, config, guardar_etapas, progreso)

def nombre_nivel(degradacion):
    for nombre, p in NIVELES_ESPECULARES.items():
        if (abs(degradacion.g - p['g']) < 1e-9 and abs(degradacion.o - p['o']) < 1e-9 and abs(degradacion.sigma_esp_px - p['sigma']) < 1e-9):
            return nombre
    return 'personalizado'

def config_con_nivel(config, nivel):
    p = NIVELES_ESPECULARES[nivel]
    cfg = copy.deepcopy(config)
    cfg.degradacion.g = p['g']
    cfg.degradacion.o = p['o']
    cfg.degradacion.sigma_esp_px = p['sigma']
    return cfg

def iterar_niveles(vertices, faces, config, niveles=None, progreso=None):
    errores = config.validar()
    if errores:
        raise ValueError('COnfiguración no válida: ' + '; '.join(errores))
    if niveles is None:
        pares = [(nombre_nivel(config.degradacion), config)]
    else:
        desconocidos = [n for n in niveles if n not in NIVELES_ESPECULARES]
        if desconocidos:
            raise ValueError(f'Niveles desconocidos: {desconocidos}')
        pares = [(n, config_con_nivel(config, n)) for n in niveles]
    geometria = calcular_geometria(vertices, faces, config, progreso)
    for nombre, cfg in pares:
        yield nombre, generar_franjas(geometria, cfg)

def verificar_escena(resultado):
    p = []
    cfg, g = resultado.config, resultado.geometria
    n, alto, ancho = len(resultado.imagenes_finales), *g.omega_cam.shape

    if n != cfg.franjas.n_pasos:
        p.append(f'se esperaban {cfg.franjas.n_pasos} imágenes y hay {n}')
    if (alto, ancho) != (cfg.camara.alto, cfg.camara.ancho):
        p.append('las dimensiones no coinciden con la cámara configurada')
    if not g.omega_map.any():
        p.append('Ω_map está vacío: no hay píxeles iluminados')
    for nombre, pila in (('ideales', resultado.imagenes_ideales), ('finales', resultado.imagenes_finales)):
        if pila.shape[1:] != (alto, ancho):
            p.append(f'imágenes {nombre}: dimensiones inconsistentes')
        if not np.isfinite(pila).all():
            p.append(f'imágenes {nombre}: hay valores no finitos')
        elif pila.min() < 0.0 or pila.max() > 1.0:
            p.append(f'imágenes {nombre}: intensidades fuera de [0, 1]')
    if not np.isfinite(resultado.fase_continua[g.omega_map]).all():
        p.append('Φ_geom no es finita dentro de Ω_map')
    if not np.isfinite(resultado.fase_ideal[g.omega_map]).all():
        p.append('φ_ideal no es finita dentro de Ω_map')
    if abs(resultado.r_ideal - 1.0) > 1e-12 and g.omega_map.any():
        p.append(f'r_ideal = {resultado.r_ideal:.6f} (debería ser 1)')
    return p