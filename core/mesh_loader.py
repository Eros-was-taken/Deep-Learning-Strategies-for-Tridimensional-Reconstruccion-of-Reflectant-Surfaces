import trimesh
import numpy as np

def load_mesh(file_path):
    try:
        loaded = trimesh.load(file_path)

        if isinstance(loaded, trimesh.Scene):
            mesh = list(loaded.geometry.values())[0]
        else:
            mesh = loaded

        info = {
            'nombre': file_path.split('/')[-1].split('\\')[-1],
            'num_puntos': len(mesh.vertices),
            'num_caras': len(mesh.faces),
            'dimension_x': round(mesh.bounds[1][0] - mesh.bounds[0][0], 4),
            'dimension_y': round(mesh.bounds[1][1] - mesh.bounds[0][1], 4),
            'dimension_z': round(mesh.bounds[1][2] - mesh.bounds[0][2], 4),
        }

        return mesh, info
    
    except Exception as e:
        raise RuntimeError(f'No se pudo cargar el archivo: {str(e)}')

def extract_point_cloud(mesh):
    vertices = np.array(mesh.vertices)

    z_values = vertices[:, 1]
    z_min = z_values.min()
    z_max = z_values.max()
    z_normalized = (z_values - z_min) / (z_max - z_min)

    return vertices, z_normalized