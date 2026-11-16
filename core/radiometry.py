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

def ideal_fringe_image(u_p, n_dot_l, omega_cam, omega_map, period_px=64.0, u0_px=512.5, k=1, n_steps=12, ambient=0.05):
    R = lambertian_response(n_dot_l, omega_map)
    I_geom = fringe_pattern(u_p, period_px, u0_px, k, n_steps)

    imagen = np.zeros(omega_cam.shape, dtype=float)
    imagen[omega_cam] = ambient
    imagen[omega_map] = R[omega_map] * I_geom[omega_map]
    return imagen, R, I_geom

def add_gaussian_noise(I_ideal, omega_map, sigma=0.01, seed=2026, k=1):
    rng = np.random.RandomState([seed, k])
    I_noise = np.zeros(I_ideal.shape, dtype=float)
    eta = rng.normal(0.0, sigma, size=int(omega_map.sum()))
    I_noise[omega_map] = np.clip(I_ideal[omega_map] + eta, 0.0, 1.0)
    return I_noise

def apply_gamma(I_noise, gamma=2.6):
    return np.power(I_noise, gamma)

def apply_noise_and_gamma(I_ideal, omega_map, sigma=0.01, gamma=2.6, seed=2026, k=1):
    I_noise = add_gaussian_noise(I_ideal, omega_map, sigma, seed, k)
    return I_noise, apply_gamma(I_noise, gamma)

NIVELES_ESPECULARES = {
    'low': {'g': 0.30, 'o': 0.03, 'sigma': 2.0},
    'medium': {'g': 0.75, 'o': 0.12, 'sigma': 6.0},
    'moderate': {'g': 1.10, 'o': 0.30, 'sigma': 9.0},
    'strong': {'g': 1.40, 'o': 0.65, 'sigma': 12.0},
    'very_strong': {'g': 1.70, 'o': 1.00, 'sigma': 16.0}
}

def specular_map(P_cam, N_cam, C_proy, omega_map, kappa=60.0):
    S = np.zeros(omega_map.shape, dtype=float)
    P = P_cam[omega_map]
    N = N_cam[omega_map]
    v = -P / np.linalg.norm(P, axis=1, keepdims=True)
    l = C_proy - P
    l = l / np.linalg.norm(l, axis=1, keepdims=True)
    h = l + v
    h = h / np.linalg.norm(h, axis=1, keepdims=True)
    S_tilde = np.maximum(0.0, np.einsum('ij,ij->i', N, h)) ** kappa
    maximo = S_tilde.max() if S_tilde.size > 0 else 0.0
    if maximo > 0.0:
        S[omega_map] = S_tilde / maximo
    return S

def gaussian_blur(imagen, sigma_px):
    r = int(np.ceil(3.0 * sigma_px))
    x = np.arange(-r, r + 1, dtype=float)
    g = np.exp(-x ** 2 / (2.0 * sigma_px ** 2))
    g /= g.sum()
    nucleo = np.outer(g, g)
    alto, ancho = imagen.shape
    forma = (alto + 2 * r, ancho + 2 * r)
    espectro = np.fft.rfft2(imagen, forma) * np.fft.rfft2(nucleo, forma)
    completa = np.fft.irfft2(espectro, forma)
    return completa[r:r + alto, r:r + ancho]

def expand_specular(S, omega_map, sigma_px):
    if sigma_px > 0:
        borroso = np.maximum(gaussian_blur(S, sigma_px), 0.0)
    else:
        borroso = S.copy()
    borroso[~omega_map] = 0.0
    maximo = borroso[omega_map].max() if omega_map.any() else 0.0
    if maximo > 0.0:
        borroso /= maximo
    return borroso

def add_specular(I_gamma, S_exp, omega_map, g, o):
    I_pre = np.zeros(I_gamma.shape, dtype=float)
    I_pre[omega_map] = (I_gamma[omega_map] * (1.0 + g * S_exp[omega_map]) + o * S_exp[omega_map])
    return I_pre

def clip_sensor(I_pre, omega_map):
    I_deg = np.zeros(I_pre.shape, dtype=float)
    I_deg[omega_map] = np.minimum(I_pre[omega_map], 1.0)
    return I_deg

def saturated_pixels(I_pre, omega_map):
    return omega_map & (I_pre >= 1.0)

def degrade_fringe_image(I_ideal, omega_map, S_exp, sigma, gamma, seed, k, g, o):
    I_ruido, I_gamma = apply_noise_and_gamma(I_ideal, omega_map, sigma, gamma, seed, k)

    I_pre = add_specular(I_gamma, S_exp, omega_map, g, o)
    return {'ideal': I_ideal, 'ruido': I_ruido, 'gamma': I_gamma, 'pre': I_pre, 'final': clip_sensor(I_pre, omega_map),}