import numpy as np

def wrap_phase(fase):
    return np.arctan2(np.sin(fase), np.cos(fase))

def geometric_phase(u_p, omega_map, period_px, u0_px):
    fase = np.full(omega_map.shape, np.nan, dtype=float)
    fase[omega_map] = 2.0 * np.pi * (u_p[omega_map] - u0_px) / period_px
    return fase

def psa_phase(imagenes, omega_map):
    pila = np.asarray(imagenes, dtype=float)
    n_pasos = pila.shape[0]
    delta = 2.0 * np.pi * np.arange(n_pasos) / n_pasos
    numerador = -np.tensordot(np.sin(delta), pila, axes=1)
    denominador = np.tensordot(np.cos(delta), pila, axes=1)
    fase = np.full(omega_map.shape, np.nan, dtype=float)
    fase[omega_map] = np.arctan2(numerador[omega_map], denominador[omega_map])
    return fase

def phase_coverage(fase_geom, omega_map):
    n_map = int(omega_map.sum())
    if n_map == 0:
        return 0.0
    return float(np.isfinite(fase_geom[omega_map]).sum()) / n_map