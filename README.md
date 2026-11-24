# Deep Learning Strategies for Tridimensional Reconstruction of Reflective Surfaces

Computational experiment on using deep learning to improve the 3D reconstruction of reflective surfaces with **Fringe Projection Profilometry (FPP)**. The synthetic dataset is built from ten geometric models and models three radiometric degradations: Gaussian noise, gamma nonlinearity and specular reflections.

> **Current state of this repository:** it contains the **fringe-projection simulator GUI** ("Simulador de Franjas"): mesh loading, camera and projector geometry, and 3D / 2D visualization. The deep learning models and the degradation pipeline are yet to be implemented.

## Project structure

```
.
├── main.py            # Entry point: launches the GUI
├── prueba.py          # Quick Open3D test: opens data/bunny.ply
├── requirements.txt   # Pinned dependencies
├── core/
│   ├── mesh_loader.py # Loads .ply meshes (trimesh) and extracts the point cloud
│   ├── camera.py      # Intrinsics/extrinsics for camera and projector, point projection
│   ├── geometry.py    # Object conditioning (rotation, centering, scaling) and system layout
│   └── view_planning.py # Fibonacci candidates, decimation, visibility and greedy selection
├── tests/
│   └── test_view_planning.py
├── benchmark_view_planning.py # Compare planning cost with exhaustive simulation
├── data/
│   └── bunny.ply      # Example model (Stanford bunny)
└── ui/
    ├── main_window.py # PySide6 main window + PyVista 3D viewport
    └── styles.py      # Qt stylesheet
```

## Requirements

- **Python 3.11 - 3.12**
- `git`
- A desktop session. The app opens a Qt window, so it will not run on a headless server.

## Installation

```bash
# 1. Clone
git clone https://github.com/Eros-was-taken/Deep-Learning-Strategies-for-Tridimensional-Reconstruccion-of-Reflectant-Surfaces.git
cd Deep-Learning-Strategies-for-Tridimensional-Reconstruccion-of-Reflectant-Surfaces

# 2. Create and activate a virtual environment
python -m venv venv
#   Windows (PowerShell):  venv\Scripts\Activate.ps1
#   Windows (cmd):         venv\Scripts\activate.bat
#   Linux / macOS:         source venv/bin/activate

# 3. Install dependencies (large: PySide6, VTK, Open3D; may take several minutes)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running

Always run from the repository root, because the imports (`core`, `ui`) are relative to it:

```bash
python main.py
```

Optional sanity check that Open3D and the sample model work:

```bash
python prueba.py
```

## Using the application

1. Click **Cargar archivo .ply** and choose a model (try `data/bunny.ply`).
2. Pick a display mode from **MODO DE VISUALIZACION**:
   - **Malla triangular**: the shaded mesh (lighting intensity, light position and smoothing are adjustable).
   - **Nube de puntos**: the vertices as a point cloud.
   - **Proyeccion camara**: the 2D pixel-space projection of the object as seen by the camera.
   - **Sistema completo**: 3D view of the object with the camera, the projector and the baseline between them.
3. Edit the parameters in the **Cámara** and **Proyector** panels, then click **Aplicar configuración**:

| Parameter | Default |
|---|---|
| Camera focal length | 12.0 mm |
| Camera pixel pitch | 4.65 µm |
| Camera resolution | 1024 × 768 |
| Camera distance | 1000 mm |
| Object rotation X / Y | 0° / 0° |
| Projector focal length | 12.0 mm |
| Projector pixel pitch | 4.50 µm |
| Projector resolution | 1024 × 768 |
| Projector distance | 1000 mm |
| Projector angle | 12° |

The panel on the left also shows the file name, number of points and faces, and the model dimensions.

### Planning a smaller set of views

In **Barrido de orientaciones**, enable **Seleccionar vistas con planificación Fibonacci** to rank a set of candidate object orientations and export only the selected views. Since the current simulator keeps the camera/projector fixed, the candidate viewpoints are represented by rotating the object. The planner:

- samples candidate orientations over a Fibonacci sphere;
- decimates the mesh by a user-selected percentage of its original triangles;
- checks camera and projector field of view, first-hit visibility, incidence angle, and specular risk at decimated face centers;
- greedily adds views with new high-quality coverage while requiring the requested overlap with a previously selected view.

The depth checks use accelerated Open3D ray casts instead of rendering full-resolution depth maps for every candidate. This is a geometric approximation: the selected views are still simulated using the original mesh and configured image resolution. Specular risk is estimated from the simulator's half-vector model; it is not a measurement of real sensor saturation.

Planning has a fixed cost, so it may not save time when only a few candidates are requested. Verify it on your mesh and settings with:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests
.\venv\Scripts\python.exe benchmark_view_planning.py --candidates 12 --max-views 4 --resolution 256
```

The benchmark reports the measured time for exhaustive candidate simulation versus planning plus full-resolution simulation of the selected views. Compare results with the same machine and mesh; do not assume planning is faster for every workload.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'ui'` / `'core'` | Run `python main.py` from the repository root. |
| `open3d` fails to install | Use Python 3.11 or 3.12. |
| `Could not load the Qt platform plugin "xcb"` (Linux) | `sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libegl1` |
| Blank or black 3D viewport | Update your graphics drivers; on virtual machines or remote desktops try a local session. |
| `Activate.ps1 cannot be loaded` (Windows) | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| "No hay puntos visibles en la configuración actual" | The object is outside the camera view: check distance, focal length and rotation values. |
