import numpy as np

def build_intrinsic_matrix(focal_mm, pixel_size_um, width, height):
    pixel_size_mm = pixel_size_um / 1000.0
    f_px = focal_mm / pixel_size_mm

    cx = (width + 1) / 2.0
    cy = (height + 1) / 2.0

    K = np.array([
        [f_px, 0, cx],
        [0, f_px, cy],
        [0, 0, 1.0]
    ], dtype=float)

    return K, f_px, cx, cy

def build_camera_extrinsic(distance_mm):
    R = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=float)

    t = np.array([[0], [0], [distance_mm]], dtype=float)

    Rt = np.hstack([R, t])
    return Rt, R, t

def build_projector_extrinsic(distance_mm, angle_deg):
    theta = np.radians(angle_deg)

    R = np.array([
        [np.cos(theta), 0, -np.sin(theta)],
        [0, -1, 0],
        [-np.sin(theta), 0, -np.cos(theta)]
    ], dtype=float)

    t = distance_mm * np.array([
        [np.sin(theta)],
        [0],
        [np.cos(theta)]
    ], dtype=float)

    Rt = np.hstack([R, t])
    return Rt, R, t

def project_points(vertices, K, Rt):
    n = len(vertices)
    ones = np.ones((n, 1))
    vertices_h = np.hstack([vertices, ones])

    points_cam = Rt @ vertices_h.T
    valid_mask = points_cam[2, :] > 0

    points_img_h = K @ points_cam
    z = points_img_h[2, :]
    z[z == 0] = 1e-10
    
    points_img = points_img_h[:2, :] / z

    return points_img.T, valid_mask

def get_camera_center(R, t):
    return (-R.T @ t).flatten()

def compute_fov(f_px, width, height):
    return width / f_px, height / f_px