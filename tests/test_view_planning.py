import unittest

import numpy as np
import trimesh

from core.config import ConfigEscena
from core.view_planning import decimate_mesh, fibonacci_orientations, plan_views


class ViewPlanningTests(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.icosphere(subdivisions=2)
        self.config = ConfigEscena()
        self.config.camara.ancho = self.config.proyector.ancho = 160
        self.config.camara.alto = self.config.proyector.alto = 120

    def test_fibonacci_orientations_are_distinct_and_map_to_unit_directions(self):
        orientations = fibonacci_orientations(16)
        self.assertEqual(len(set(orientations)), 16)

        for alpha, beta in orientations:
            alpha_rad, beta_rad = np.radians([alpha, beta])
            direction = np.array([
                np.sin(beta_rad) * np.cos(alpha_rad),
                -np.sin(alpha_rad),
                np.cos(beta_rad) * np.cos(alpha_rad),
            ])
            self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=6)

    def test_decimation_percentage_controls_retained_face_count(self):
        _, reduced_faces = decimate_mesh(
            self.mesh.vertices, self.mesh.faces, percentage=25.0
        )
        self.assertGreater(len(reduced_faces), 0)
        self.assertLess(len(reduced_faces), len(self.mesh.faces))
        self.assertAlmostEqual(
            len(reduced_faces) / len(self.mesh.faces), 0.25, delta=0.08
        )

    def test_planner_returns_a_smaller_measurable_view_set(self):
        plan = plan_views(
            self.mesh.vertices,
            self.mesh.faces,
            self.config,
            candidate_count=12,
            max_views=3,
            decimation_percentage=50.0,
            min_overlap=0.0,
        )

        self.assertGreater(len(plan.orientaciones), 0)
        self.assertLessEqual(len(plan.orientaciones), 3)
        self.assertLess(len(plan.orientaciones), 12)
        self.assertEqual(plan.total_caras, len(self.mesh.faces))
        self.assertGreater(plan.caras_decimadas, 0)
        self.assertGreater(plan.cobertura, 0.0)
        self.assertLessEqual(plan.cobertura, 1.0)
        self.assertTrue(np.all(plan.calidad_cubierta >= 0.0))

    def test_invalid_decimation_percentage_is_rejected(self):
        with self.assertRaises(ValueError):
            decimate_mesh(self.mesh.vertices, self.mesh.faces, percentage=0.0)


if __name__ == '__main__':
    unittest.main()
