import numpy as np

def rotation_matrix_x(angle_deg):
    angle = np.radians(angle_deg)
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ], dtype=float)

def rotation_matrix_y(angle_deg):
    angle = np.radians(angle_deg)
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ], dtype=float)

def condition_mesh(vertices, alpha_deg, beta_deg, f_px, width, height, distance_mm):
    Rx = rotation_matrix_x(alpha_deg)
    Ry = rotation_matrix_y(beta_deg)
    R_orient = Ry @ Rx

    vertices_rot = (R_orient @ vertices.T).T

    bbox_min = vertices_rot.min(axis=0)
    bbox_max = vertices_rot.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.0

    vertices_centered = vertices_rot - bbox_center

    Lx = (width / f_px) * distance_mm
    Ly = (height / f_px) * distance_mm

    delta_x = bbox_max[0] - bbox_min[0]
    delta_y = bbox_max[1] - bbox_min[1]

    if delta_x == 0 or delta_y == 0:
        scale = 1.0
    else:
        scale = min(0.90 * Lx / delta_x, 0.90 * Ly / delta_y)

    vertices_scaled = vertices_centered * scale

    R_cam = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ], dtype=float)

    vertices_cam = (R_cam @ vertices_scaled.T).T
    vertices_cam[:, 2] += distance_mm

    return vertices_cam, scale, Lx, Ly