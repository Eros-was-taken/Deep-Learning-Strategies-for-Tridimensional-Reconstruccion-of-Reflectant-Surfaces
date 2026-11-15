import numpy as np

def project_vertices(vertices_cam, K):
    x = vertices_cam[:, 0]
    y = vertices_cam[:, 1]
    z = vertices_cam[:, 2]

    z_safe = np.where(z > 1e-9, z, 1.0)

    u = K[0, 0] * x / z_safe + K[0, 2]
    v = K[1, 1] * y / z_safe + K[1, 2]
    return u, v, z

def rasterize_zbuffer(vertices_cam, faces, K, width, height):
    faces = np.asarray(faces, dtype=np.int64)
    u, v, z = project_vertices(vertices_cam, K)

    U, V, Z = u[faces], v[faces], z[faces]

    det = (V[:, 1] - V[:, 2]) * (U[:, 0] - U[:, 2]) + (U[:, 2] - U[:, 1]) * (V[:, 0] - V[:, 2])

    u_min = np.maximum(np.ceil(U.min(axis=1)), 1).astype(np.int64)
    u_max = np.minimum(np.floor(U.max(axis=1)), width).astype(np.int64)
    v_min = np.maximum(np.ceil(V.min(axis=1)), 1).astype(np.int64)
    v_max = np.minimum(np.floor(V.max(axis=1)), height).astype(np.int64)

    validas = ((Z.min(axis=1) > 1e-9) & (np.abs(det) > 1e-12) & (u_min <= u_max) & (v_min <= v_max))

    depth = np.full((height, width), np.inf)
    face_id = np.full((height, width), -1, dtype=np.int64)
    eps = 1e-9

    for f in np.flatnonzero(validas):
        u0, u1, u2 = U[f]
        v0, v1, v2 = V[f]
        z0, z1, z2 = Z[f]

        uu, vv = np.meshgrid(np.arange(u_min[f], u_max[f] + 1), np.arange(v_min[f], v_max[f] + 1))

        l0 = ((v1 - v2) * (uu - u2) + (u2 - u1) * (vv - v2)) / det[f]
        l1 = ((v2 - v0) * (uu - u2) + (u0 - u2) * (vv - v2)) / det[f]
        l2 = 1.0 - l0 - l1

        dentro = (l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps)
        if not dentro.any():
            continue

        z_pix = 1.0 / (l0 / z0 + l1 / z1 + l2 / z2)

        sub_depth = depth[v_min[f] - 1:v_max[f], u_min[f] - 1:u_max[f]]
        sub_face = face_id[v_min[f] - 1:v_max[f], u_min[f] - 1:u_max[f]]

        actualizar = dentro & (z_pix < sub_depth)
        sub_depth[actualizar] = z_pix[actualizar]
        sub_face[actualizar] = f
    
    omega_cam = face_id >= 0
    return depth, face_id, omega_cam

def compute_point_map_and_normals(vertices_cam, faces, face_id, depth, K):
    faces = np.asarray(faces, dtype=np.int64)
    alto, ancho = depth.shape
    omega = face_id >= 0

    uu, vv = np.meshgrid(np.arange(1, ancho + 1), np.arange(1, alto + 1))
    P = np.full((alto, ancho, 3), np.nan)
    z = depth[omega]
    P[omega, 0] = (uu[omega] - K[0, 2]) * z / K[0, 0]
    P[omega, 1] = (vv[omega] - K[1, 2]) * z / K[1, 1]
    P[omega, 2] = z

    A, B, C = (vertices_cam[faces[:, i]] for i in range(3))
    n_tri = np.cross(B - A, C - A)
    norma = np.linalg.norm(n_tri, axis=1, keepdims=True)
    n_tri = n_tri / np.where(norma > 1e-15, norma, 1.0)

    N = np.full((alto, ancho, 3), np.nan)
    n_pix = n_tri[face_id[omega]]
    signo = np.where(np.einsum('ij,ij->i', n_pix, P[omega]) > 0, -1.0, 1.0)
    N[omega] = n_pix * signo[:, None]
    return P, N

def camera_to_projector_coords(P_cam, R_po, t_po, K_p, distance_cam_mm):
    R_oc = np.diag([1.0, -1.0, -1.0])
    T_oc = np.array([0.0, 0.0, distance_cam_mm])
    R_po = np.asarray(R_po, dtype=float).reshape(3, 3)
    t_po = np.asarray(t_po, dtype=float).reshape(3)

    X_obj = (P_cam - T_oc) @ R_oc
    X_proj = (X_obj - t_po) @ R_po

    x, y, z = X_proj[..., 0], X_proj[..., 1], X_proj[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u_p = K_p[0, 0] * x / z + K_p[0, 2]
        v_p = K_p[1, 1] * y / z + K_p[1, 2]
    return u_p, v_p, z

# Visibilidad desde el proyector
ETQ_FUERA_CAMARA = 0
ETQ_ILUMINADO = 1
ETQ_FUERA_CAMPO = 2
ETQ_SOMBRA_PROPIA = 3
ETQ_SOMBRA_PROYECTADA = 4

def compute_projector_visibility(vertices_cam, faces, P_cam, N_cam, omega_cam, R_po, t_po, K_p, distance_cam_mm, width_p, height_p, bias_mm=1.5):
    R_oc = np.diag([1.0, -1.0, -1.0])
    T_oc = np.array([0.0, 0.0, distance_cam_mm])
    R_po = np.asarray(R_po, dtype=float).reshape(3, 3)
    t_po = np.asarray(t_po, dtype=float).reshape(3)

    X_obj = (vertices_cam - T_oc) @ R_oc
    V_proj = (X_obj - t_po) @ R_po
    depth_p, _, _ = rasterize_zbuffer(V_proj, faces, K_p, width_p, height_p)

    u_p, v_p, z_p = camera_to_projector_coords(P_cam, R_po, t_po, K_p, distance_cam_mm)

    with np.errstate(invalid='ignore'):
        iu = np.rint(u_p)
        iv = np.rint(v_p)
    en_campo = (omega_cam & np.isfinite(iu) & np.isfinite(iv) & (iu >= 1) & (iu <= width_p) & (iv >= 1) & (iv <= height_p))

    C_proy = R_oc @ t_po + T_oc
    L = C_proy - P_cam
    L = L / np.linalg.norm(L, axis=-1, keepdims=True)
    n_dot_l = np.einsum('ijk,ijk->ij', N_cam, L)
    mira_al_proyector = n_dot_l > 1e-6

    iu_i = np.where(en_campo, iu, 1).astype(np.int64) - 1
    iv_i = np.where(en_campo, iv, 1).astype(np.int64) - 1
    d_proy = depth_p[iv_i, iu_i]
    sin_oclusion = z_p <= d_proy + bias_mm

    etiquetas = np.full(omega_cam.shape, ETQ_FUERA_CAMARA, dtype=np.uint8)
    etiquetas[omega_cam] = ETQ_ILUMINADO
    etiquetas[omega_cam & ~en_campo] = ETQ_FUERA_CAMPO
    etiquetas[omega_cam & ~mira_al_proyector] = ETQ_SOMBRA_PROPIA
    etiquetas[omega_cam & mira_al_proyector & ~sin_oclusion] = ETQ_SOMBRA_PROYECTADA

    info = {'u_p': u_p, 'v_p': v_p, 'z_p': z_p, 'depth_proj': depth_p, 'n_dot_l': n_dot_l, 'C_proy': C_proy}
    return etiquetas, info