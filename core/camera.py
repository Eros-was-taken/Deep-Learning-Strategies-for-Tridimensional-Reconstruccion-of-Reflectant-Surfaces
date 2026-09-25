import numpy as np

def build_intrinsic_matrix(focal_length, cx, cy):
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=float)
    return K

def build_extrinsic_matrix(translation, rotation_deg):
    rx, ry, rz = np.radians(rotation_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    t = np.array(translation, dtype=float).reshape(3, 1)

    Rt = np.hstack([R, t])
    return Rt

def project_points(vertices, K, Rt):
    n = len(vertices)
    ones = np.ones((n, 1))
    vertices_h = np.hstack([vertices, ones])

    points_cam = (Rt @ vertices_h.T)

    valid_mask = points_cam[2, :] > 0

    points_img_h = K @ points_cam
    
    points_img = points_img_h[:2, :] / points_img_h[2, :]

    return points_img.T, valid_mask