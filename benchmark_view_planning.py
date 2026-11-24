"""Compare exhaustive candidate simulation with planning plus selected views."""

import argparse
import copy
from time import perf_counter

from core.config import ConfigEscena
from core.mesh_loader import load_mesh
from core.simulator import simular_escena
from core.view_planning import fibonacci_orientations, plan_views


def _simulate_views(vertices, faces, base_config, orientations):
    started = perf_counter()
    for alpha, beta in orientations:
        config = copy.deepcopy(base_config)
        config.objeto.alpha_deg = alpha
        config.objeto.beta_deg = beta
        simular_escena(vertices, faces, config)
    return perf_counter() - started


def main():
    parser = argparse.ArgumentParser(
        description='Mide el coste de planificación frente a simular todos los candidatos.'
    )
    parser.add_argument('--mesh', default='data/bunny.ply')
    parser.add_argument('--candidates', type=int, default=12)
    parser.add_argument('--max-views', type=int, default=4)
    parser.add_argument('--decimation', type=float, default=10.0,
                        help='Porcentaje de caras conservadas para planificar.')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Ancho de cámara/proyector usado en la comparación.')
    args = parser.parse_args()
    if args.candidates < 1 or args.max_views < 1 or args.resolution < 32:
        parser.error('candidates y max-views deben ser positivos; resolution >= 32.')

    mesh, _ = load_mesh(args.mesh)
    config = ConfigEscena()
    config.camara.ancho = config.proyector.ancho = args.resolution
    config.camara.alto = config.proyector.alto = round(args.resolution * 0.75)
    config.franjas.n_pasos = 3
    candidates = fibonacci_orientations(args.candidates)

    exhaustive_seconds = _simulate_views(
        mesh.vertices, mesh.faces, config, candidates
    )
    plan = plan_views(
        mesh.vertices,
        mesh.faces,
        config,
        candidate_count=args.candidates,
        max_views=args.max_views,
        decimation_percentage=args.decimation,
    )
    selected_seconds = _simulate_views(
        mesh.vertices, mesh.faces, config, plan.orientaciones
    )
    planned_total = plan.tiempo_segundos + selected_seconds
    saved = 1.0 - planned_total / exhaustive_seconds if exhaustive_seconds else 0.0

    print(f'Exhaustive ({args.candidates} full-resolution views): {exhaustive_seconds:.2f} s')
    print(
        f'Planning ({args.candidates} candidates, {plan.caras_decimadas} decimated '
        f'faces): {plan.tiempo_segundos:.2f} s'
    )
    print(f'Selected full-resolution views ({len(plan.orientaciones)}): {selected_seconds:.2f} s')
    print(f'Planning + selected simulations: {planned_total:.2f} s')
    print(f'Estimated time saved: {saved:.1%}')
    print(f'Estimated face coverage: {plan.cobertura:.1%}')
    if saved <= 0.0:
        print('For these settings, planning overhead exceeds the saved simulation time.')


if __name__ == '__main__':
    main()
