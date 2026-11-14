import numpy as np

def lambertian_response(n_dot_l, omega_map):
    R = np.zeros(n_dot_l.shape, dtype=float)
    R_tilde = np.maximum(0.0, n_dot_l[omega_map])
    maximo = R_tilde.max() if R_tilde.size > 0 else 0.0
    if maximo > 0.0:
        R[omega_map] = R_tilde / maximo
    return R

def fringe_pattern(u_p, period_px, u0_px, k, n_steps):
    delta_k = 2.0 * np.pi * (k - 1) / n_steps
    return 0.5 + 0.5 * np.cos(2.0 * np.pi * (u_p -u0_px) / period_px + delta_k)

def ideal_fringe_image(u_p, n_dot_l, omega_cam, omega_map, period_px=64.0, u0_px=1.0, k=1, n_steps=12, ambient=0.05):
    R = lambertian_response(n_dot_l, omega_map)
    I_geom = fringe_pattern(u_p, period_px, u0_px, k, n_steps)

    imagen = np.zeros(omega_cam.shape, dtype=float)
    imagen[omega_cam] = ambient
    imagen[omega_map] = R[omega_map] * I_geom[omega_map]
    return imagen, R, I_geom