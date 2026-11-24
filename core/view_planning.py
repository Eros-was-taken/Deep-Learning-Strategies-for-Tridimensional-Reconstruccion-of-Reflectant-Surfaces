"""Fibonacci-sphere view planning for the fixed camera/projector simulator."""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from core.camera import (
    build_intrinsic_matrix,
    build_projector_extrinsic,
)
from core.config import ConfigEscena
from core.geometry import condition_mesh
from core.visibility import camera_to_projector_coords


@dataclass
class ViewPlan:
    """Selected object orientations and the quality estimated for each mesh face."""

    orientaciones: list[tuple[float, float]]
    caras_cubiertas: int
    total_caras: int
    caras_decimadas: int
    tiempo_segundos: float
    calidad_cubierta: np.ndarray

    @property
    def cobertura(self) -> float:
        """Fraction of original faces estimated measurable in at least one view."""
        return self.caras_cubiertas / self.total_caras if self.total_caras else 0.0


@dataclass
class _Candidato:
    orientacion: tuple[float, float]
    calidad_por_cara: np.ndarray


def fibonacci_orientations(count: int) -> list[tuple[float, float]]:
    """Map evenly distributed Fibonacci-sphere directions to the simulator's X/Y tilts."""
    if count < 1:
        raise ValueError('La cantidad de vistas candidatas debe ser al menos 1.')

    indices = np.arange(count, dtype=float)
    y = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    azimuth = np.pi * (3.0 - np.sqrt(5.0)) * indices
    x = radius * np.cos(azimuth)
    z = radius * np.sin(azimuth)

    # R_y(beta) @ R_x(alpha) sends the original +Z direction to (x, y, z).
    alpha = np.degrees(np.arcsin(np.clip(-y, -1.0, 1.0)))
    beta = np.degrees(np.arctan2(x, z))
    return [(round(float(a), 6), round(float(b), 6)) for a, b in zip(alpha, beta)]


def decimate_mesh(
    vertices: np.ndarray, faces: np.ndarray, percentage: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep approximately `percentage` percent of the original triangles."""
    if not 0.0 < percentage <= 100.0:
        raise ValueError('El porcentaje de malla debe estar en el intervalo (0, 100].')

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError('La planificación requiere una malla triangular no vacía.')

    target_faces = max(4, round(len(faces) * percentage / 100.0))
    if target_faces >= len(faces):
        return vertices.copy(), faces.copy()

    # Open3D's quadric simplifier is already a project dependency and preserves
    # the mesh surface better than selecting an arbitrary subset of triangles.
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces),
    )
    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    reduced_vertices = np.asarray(simplified.vertices, dtype=float)
    reduced_faces = np.asarray(simplified.triangles, dtype=np.int64)
    if len(reduced_faces) == 0:
        raise RuntimeError('La decimación no produjo caras; aumenta el porcentaje de malla.')
    return reduced_vertices, reduced_faces


def _quality_per_face(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: ConfigEscena,
    orientation: tuple[float, float],
    max_incidence_deg: float,
    specular_risk_threshold: float,
) -> np.ndarray:
    camera = config.camara
    vertices_cam, _, _, _ = condition_mesh(
        vertices,
        orientation[0],
        orientation[1],
        camera.focal_mm / (camera.pixel_um / 1000.0),
        camera.ancho,
        camera.alto,
        camera.distancia_mm,
    )

    triangles = vertices_cam[faces]
    points = triangles.mean(axis=1)
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normal_lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(normal_lengths[:, None], 1e-12)
    camera_directions = -points
    camera_distances = np.linalg.norm(camera_directions, axis=1)
    camera_directions /= np.maximum(camera_distances[:, None], 1e-12)
    normals *= np.where(
        np.einsum('ij,ij->i', normals, camera_directions) < 0.0, -1.0, 1.0
    )[:, None]

    K_cam, _, _, _ = build_intrinsic_matrix(
        camera.focal_mm, camera.pixel_um, camera.ancho, camera.alto
    )
    u_cam = K_cam[0, 0] * points[:, 0] / points[:, 2] + K_cam[0, 2]
    v_cam = K_cam[1, 1] * points[:, 1] / points[:, 2] + K_cam[1, 2]
    in_camera_frame = (
        (points[:, 2] > 0.0)
        & (u_cam >= 1.0)
        & (u_cam <= camera.ancho)
        & (v_cam >= 1.0)
        & (v_cam <= camera.alto)
    )

    _, R_projector, t_projector = build_projector_extrinsic(
        config.proyector.distancia_mm, config.proyector.angulo_deg
    )
    K_projector, _, _, _ = build_intrinsic_matrix(
        config.proyector.focal_mm,
        config.proyector.pixel_um,
        config.proyector.ancho,
        config.proyector.alto,
    )
    u_projector, v_projector, z_projector = camera_to_projector_coords(
        points,
        R_projector,
        t_projector,
        K_projector,
        camera.distancia_mm,
    )
    in_projector_frame = (
        (z_projector > 0.0)
        & (u_projector >= 1.0)
        & (u_projector <= config.proyector.ancho)
        & (v_projector >= 1.0)
        & (v_projector <= config.proyector.alto)
    )

    projector_center = (
        np.diag([1.0, -1.0, -1.0]) @ np.asarray(t_projector).reshape(3)
        + np.array([0.0, 0.0, camera.distancia_mm])
    )
    projector_directions = projector_center - points
    projector_distances = np.linalg.norm(projector_directions, axis=1)
    projector_directions /= np.maximum(projector_distances[:, None], 1e-12)

    # One center ray per decimated face approximates the depth-map visibility
    # test without rasterizing thousands of pixels for every candidate pose.
    import open3d as o3d

    triangle_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices_cam),
        o3d.utility.Vector3iVector(faces.astype(np.int32, copy=False)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(triangle_mesh))
    epsilon = max(float(np.max(np.ptp(vertices_cam, axis=0))) * 1e-5, 1e-4)

    camera_rays = np.concatenate(
        (np.zeros_like(points), -camera_directions), axis=1
    ).astype(np.float32)
    camera_hits = scene.cast_rays(o3d.core.Tensor(camera_rays))['t_hit'].numpy()
    camera_visible = np.isfinite(camera_hits) & (
        np.abs(camera_hits - camera_distances) <= epsilon
    )
    projector_rays = np.concatenate(
        (np.broadcast_to(projector_center, points.shape), -projector_directions), axis=1
    ).astype(np.float32)
    projector_hits = scene.cast_rays(o3d.core.Tensor(projector_rays))['t_hit'].numpy()
    projector_visible = np.isfinite(projector_hits) & (
        np.abs(projector_hits - projector_distances) <= epsilon
    )

    cosine_camera = np.clip(
        np.einsum('ij,ij->i', normals, camera_directions), 0.0, 1.0
    )
    cosine_projector = np.clip(
        np.einsum('ij,ij->i', normals, projector_directions), 0.0, 1.0
    )
    incidence_ok = cosine_camera >= np.cos(np.radians(max_incidence_deg))
    incidence_ok &= cosine_projector >= np.cos(np.radians(max_incidence_deg))

    half_vector = projector_directions + camera_directions
    half_vector /= np.maximum(np.linalg.norm(half_vector, axis=1, keepdims=True), 1e-12)
    specular = np.maximum(0.0, np.einsum('ij,ij->i', normals, half_vector)) ** (
        config.degradacion.kappa
    )
    measurable = (
        in_camera_frame
        & in_projector_frame
        & camera_visible
        & projector_visible
        & (cosine_projector > 0.0)
    )
    peak = specular[measurable].max() if measurable.any() else 0.0
    if peak > 0.0:
        specular /= peak
    safe_from_glare = specular < specular_risk_threshold

    accepted = measurable & incidence_ok & safe_from_glare
    quality = np.zeros(len(faces), dtype=float)
    quality[accepted] = (
        cosine_camera[accepted]
        * cosine_projector[accepted]
        * (1.0 - specular[accepted])
    )
    return quality


def plan_views(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: ConfigEscena,
    candidate_count: int = 48,
    max_views: int = 8,
    decimation_percentage: float = 10.0,
    max_incidence_deg: float = 60.0,
    min_overlap: float = 0.15,
    specular_risk_threshold: float = 0.85,
    progreso: Callable[[int, int, str], None] | None = None,
    cancelar: Callable[[], bool] | None = None,
) -> ViewPlan:
    """Greedily select high-quality Fibonacci views while retaining overlap.

    Quality is evaluated at decimated face centers. Camera and projector
    ray-casts provide first-hit depth tests without rendering fringe images.
    The selected orientations are later simulated with the original mesh and
    full configured image resolution.
    """
    if candidate_count < 1 or max_views < 1:
        raise ValueError('La cantidad de candidatas y de vistas debe ser al menos 1.')
    if not 0.0 <= max_incidence_deg <= 90.0:
        raise ValueError('El ángulo máximo de incidencia debe estar entre 0° y 90°.')
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError('El solapamiento mínimo debe estar entre 0 y 1.')
    if not 0.0 < specular_risk_threshold <= 1.0:
        raise ValueError('El umbral de riesgo especular debe estar en (0, 1].')
    config_errors = config.validar()
    if config_errors:
        raise ValueError('Configuración no válida: ' + '; '.join(config_errors))

    started = perf_counter()
    reduced_vertices, reduced_faces = decimate_mesh(
        vertices, faces, decimation_percentage
    )
    orientations = fibonacci_orientations(candidate_count)
    candidates = []
    for index, orientation in enumerate(orientations, start=1):
        if cancelar is not None and cancelar():
            raise InterruptedError('Planificación de vistas cancelada.')
        quality = _quality_per_face(
            reduced_vertices,
            reduced_faces,
            config,
            orientation,
            max_incidence_deg,
            specular_risk_threshold,
        )
        if np.any(quality > 0.0):
            candidates.append(_Candidato(orientation, quality))
        if progreso is not None:
            progreso(
                index,
                candidate_count,
                f'Analizando vista candidata {index} de {candidate_count}',
            )

    if not candidates:
        raise ValueError('Ninguna vista candidata permite medir caras de la malla.')

    # Greedy max-quality coverage: a face's current quality is the best among
    # selected views, so redundant views have diminishing marginal benefit.
    covered_quality = np.zeros(len(reduced_faces), dtype=float)
    selected: list[_Candidato] = []
    remaining = list(candidates)
    while remaining and len(selected) < max_views:
        eligible = remaining
        if selected and min_overlap > 0.0:
            eligible = [
                candidate
                for candidate in remaining
                if max(
                    _overlap(candidate.calidad_por_cara, chosen.calidad_por_cara)
                    for chosen in selected
                )
                >= min_overlap
            ]
        if not eligible:
            break
        best = max(
            eligible,
            key=lambda candidate: float(
                np.maximum(candidate.calidad_por_cara - covered_quality, 0.0).sum()
            ),
        )
        gain = np.maximum(best.calidad_por_cara - covered_quality, 0.0).sum()
        if gain <= 1e-12:
            break
        selected.append(best)
        remaining.remove(best)
        covered_quality = np.maximum(covered_quality, best.calidad_por_cara)

    if not selected:
        raise ValueError('No fue posible seleccionar una vista medible.')

    # Extrapolate the covered-face fraction on the decimated mesh to the
    # original face count for a useful, explicitly approximate UI estimate.
    covered_decimated = int(np.count_nonzero(covered_quality > 0.0))
    original_face_count = len(np.asarray(faces))
    estimated_covered = round(
        covered_decimated / len(reduced_faces) * original_face_count
    )
    return ViewPlan(
        orientaciones=[candidate.orientacion for candidate in selected],
        caras_cubiertas=estimated_covered,
        total_caras=original_face_count,
        caras_decimadas=len(reduced_faces),
        tiempo_segundos=perf_counter() - started,
        calidad_cubierta=covered_quality,
    )


def _overlap(first: np.ndarray, second: np.ndarray) -> float:
    """Intersection over the smaller visible-face set, allowing adjacent views."""
    first_visible = first > 0.0
    second_visible = second > 0.0
    denominator = min(int(first_visible.sum()), int(second_visible.sum()))
    if denominator == 0:
        return 0.0
    return int(np.count_nonzero(first_visible & second_visible)) / denominator
