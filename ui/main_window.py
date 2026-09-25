import sys
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QFrame, QComboBox,QSlider, QCheckBox
)
from PySide6.QtCore import Qt
import pyvista as pv
from pyvistaqt import QtInteractor
import numpy as np
from core.mesh_loader import load_mesh, extract_point_cloud
from core.camera import build_intrinsic_matrix, build_extrinsic_matrix, project_points
from ui.styles import STYLE

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Simulador de Franjas')
        self.setMinimumSize(1400, 800)
        self.resize(1400, 800)
        self.showMaximized()
        self.setStyleSheet(STYLE)

        self.mesh = None
        self.mesh_info = None
        self.modo_visualizacion = 'malla'
        self.focal_length = 800
        self.resolucion = (640, 480)
        self.traduccion = [0, 0, 3]
        self.rotacion = [0, 0, 0]
        
        self._build_ui()

    def _build_ui(self):
        widget_central = QWidget()
        self.setCentralWidget(widget_central)

        layout_principal = QHBoxLayout(widget_central)
        layout_principal.setSpacing(10)
        layout_principal.setContentsMargins(10, 10, 10, 10)

        # Panel izquierdo
        panel_izquierdo = QFrame()
        panel_izquierdo.setFrameShape(QFrame.Shape.StyledPanel)
        panel_izquierdo.setFixedWidth(260)
        layout_izquierdo = QVBoxLayout(panel_izquierdo)
        layout_izquierdo.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout_izquierdo.setSpacing(12)
        layout_izquierdo.setContentsMargins(16, 16, 16, 16)

        # Titulo del panel
        titulo_panel = QLabel('MODELO 3D')
        titulo_panel.setObjectName('label_titulo')

        # Boton para cargar archivo
        self.btn_cargar = QPushButton('Cargar archivo .ply')
        self.btn_cargar.setFixedHeight(40)
        self.btn_cargar.clicked.connect(self._cargar_archivo)

        # Separador
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setFixedHeight(1)

        # Modo de visualizacion
        titulo_modo = QLabel('MODO DE VISUALIZACION')
        titulo_modo.setObjectName('label_titulo')
        self.combo_modo = QComboBox()
        self.combo_modo.addItems(['Malla triangular', 'Nube de puntos', 'Proyeccion camara'])
        self.combo_modo.currentIndexChanged.connect(self._cambiar_modo)

        # Separador
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFixedHeight(1)

        # Iluminacion
        titulo_luz = QLabel('ILUMINACION')
        titulo_luz.setObjectName('label_titulo')

        self.label_intensidad = QLabel('Intensidad: 50%')
        self.label_intensidad.setObjectName('label_valor')
        self.slider_intensidad = QSlider(Qt.Orientation.Horizontal)
        self.slider_intensidad.setMinimum(0)
        self.slider_intensidad.setMaximum(100)
        self.slider_intensidad.setValue(50)
        self.slider_intensidad.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_intensidad.setTickInterval(25)
        self.slider_intensidad.valueChanged.connect(self._actualizar_label_intensidad)
        self.slider_intensidad.valueChanged.connect(self._actualizar_visualizacion)

        titulo_posicion = QLabel('Posicion de luz')
        titulo_posicion.setObjectName('label_valor')
        self.combo_posicion = QComboBox()
        self.combo_posicion.addItems(['Superior', 'Frontal', 'Lateral'])
        self.combo_posicion.currentIndexChanged.connect(self._actualizar_visualizacion)

        # Suavizado
        self.check_suavizado = QCheckBox('Suavizado de superficie')
        self.check_suavizado.setChecked(True)
        self.check_suavizado.stateChanged.connect(self._actualizar_visualizacion)

        # Separador
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setFixedHeight(1)

        # Etiquetas de informacion
        titulo_info = QLabel('INFORMACION')
        titulo_info.setObjectName('label_titulo')

        self.label_nombre_titulo = QLabel('ARCHIVO')
        self.label_nombre_titulo.setObjectName('label_titulo')
        self.label_nombre = QLabel('-')
        self.label_nombre.setObjectName('label_valor')

        self.label_puntos_titulo = QLabel('PUNTOS')
        self.label_puntos_titulo.setObjectName('label_titulo')
        self.label_puntos = QLabel('-')
        self.label_puntos.setObjectName('label_valor')

        self.label_caras_titulo = QLabel('CARAS')
        self.label_caras_titulo.setObjectName('label_titulo')
        self.label_caras = QLabel('-')
        self.label_caras.setObjectName('label_valor')

        self.label_dim_titulo = QLabel('DIMENSIONES')
        self.label_dim_titulo.setObjectName('label_titulo')
        self.label_dim = QLabel('-')
        self.label_dim.setObjectName('label_valor')
        self.label_dim.setWordWrap(True)

        # Separador camara
        sep4 = QFrame()
        sep4.setFrameShape(QFrame.Shape.HLine)
        sep4.setFixedHeight(1)

        # Seccion camara
        titulo_camara = QLabel('CAMARA PINHOLE')
        titulo_camara.setObjectName('label_titulo')

        # Distancia focal
        self.label_focal = QLabel('Distancia focal: 800px')
        self.label_focal.setObjectName('label_valor')
        self.slider_focal = QSlider(Qt.Orientation.Horizontal)
        self.slider_focal.setMinimum(100)
        self.slider_focal.setMaximum(2000)
        self.slider_focal.setValue(800)
        self.slider_focal.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_focal.setTickInterval(500)
        self.slider_focal.valueChanged.connect(self._actualizar_label_focal)
        self.slider_focal.valueChanged.connect(self._actualizar_visualizacion)

        # Traslacion Z
        self.label_tz = QLabel('Distancia al objeto: 3')
        self.label_tz.setObjectName('label_valor')
        self.slider_tz = QSlider(Qt.Orientation.Horizontal)
        self.slider_tz.setMinimum(1)
        self.slider_tz.setMaximum(10)
        self.slider_tz.setValue(3)
        self.slider_tz.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_tz.setTickInterval(3)
        self.slider_tz.valueChanged.connect(self._actualizar_label_tz)
        self.slider_tz.valueChanged.connect(self._actualizar_visualizacion)

        # Rotacion X
        self.label_rx = QLabel('Rotacion X: 0°')
        self.label_rx.setObjectName('label_valor')
        self.slider_rx = QSlider(Qt.Orientation.Horizontal)
        self.slider_rx.setMinimum(-180)
        self.slider_rx.setMaximum(180)
        self.slider_rx.setValue(0)
        self.slider_rx.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_rx.setTickInterval(90)
        self.slider_rx.valueChanged.connect(self._actualizar_label_rx)
        self.slider_rx.valueChanged.connect(self._actualizar_visualizacion)

        # Rotacion Y
        self.label_ry = QLabel('Rotacion Y: 0°')
        self.label_ry.setObjectName('label_valor')
        self.slider_ry = QSlider(Qt.Orientation.Horizontal)
        self.slider_ry.setMinimum(-180)
        self.slider_ry.setMaximum(180)
        self.slider_ry.setValue(0)
        self.slider_ry.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_ry.setTickInterval(90)
        self.slider_ry.valueChanged.connect(self._actualizar_label_ry)
        self.slider_ry.valueChanged.connect(self._actualizar_visualizacion)

        # Agregar elementos al panel izquierdo
        layout_izquierdo.addWidget(titulo_panel)
        layout_izquierdo.addWidget(self.btn_cargar)
        layout_izquierdo.addWidget(sep1)
        layout_izquierdo.addWidget(titulo_modo)
        layout_izquierdo.addWidget(self.combo_modo)
        layout_izquierdo.addWidget(sep2)
        layout_izquierdo.addWidget(titulo_luz)
        layout_izquierdo.addWidget(self.label_intensidad)
        layout_izquierdo.addWidget(self.slider_intensidad)
        layout_izquierdo.addWidget(titulo_posicion)
        layout_izquierdo.addWidget(self.combo_posicion)
        layout_izquierdo.addWidget(self.check_suavizado)
        layout_izquierdo.addWidget(sep3)
        layout_izquierdo.addWidget(titulo_info)
        layout_izquierdo.addWidget(self.label_nombre_titulo)
        layout_izquierdo.addWidget(self.label_nombre)
        layout_izquierdo.addWidget(self.label_puntos_titulo)
        layout_izquierdo.addWidget(self.label_puntos)
        layout_izquierdo.addWidget(self.label_caras_titulo)
        layout_izquierdo.addWidget(self.label_caras)
        layout_izquierdo.addWidget(self.label_dim_titulo)
        layout_izquierdo.addWidget(self.label_dim)
        layout_izquierdo.addWidget(sep4)
        layout_izquierdo.addWidget(titulo_camara)
        layout_izquierdo.addWidget(self.label_focal)
        layout_izquierdo.addWidget(self.slider_focal)
        layout_izquierdo.addWidget(self.label_tz)
        layout_izquierdo.addWidget(self.slider_tz)
        layout_izquierdo.addWidget(self.label_rx)
        layout_izquierdo.addWidget(self.slider_rx)
        layout_izquierdo.addWidget(self.label_ry)
        layout_izquierdo.addWidget(self.slider_ry)

        # Panel derecho - Visualizacion 3D
        self.plotter = QtInteractor(self)

        layout_principal.addWidget(panel_izquierdo)
        layout_principal.addWidget(self.plotter)

    def _cargar_archivo(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Seleccionar archivo PLY',
            '',
            'Archivos PLY (*.ply)'
        )

        if not file_path:
            return

        try:
            self.mesh, self.mesh_info = load_mesh(file_path)
            self._actualizar_info()
            self._actualizar_visualizacion()

        except RuntimeError as e:
            QMessageBox.critical(self, 'Error', str(e))
    
    def _cambiar_modo(self):
        if self.combo_modo.currentIndex() == 0:
            self.modo_visualizacion = 'malla'
        elif self.combo_modo.currentIndex() == 1:
            self.modo_visualizacion = 'nube'
        else:
            self.modo_visualizacion = 'proyeccion'

        if self.mesh is not None:
            self._actualizar_visualizacion()
        
    def _actualizar_info(self):
        info = self.mesh_info
        self.label_nombre.setText(info['nombre'])
        self.label_puntos.setText(f"{info['num_puntos']:,}")
        self.label_caras.setText(f"{info['num_caras']:,}")
        self.label_dim.setText(
            f"X: {info['dimension_x']}\n"
            f"Y: {info['dimension_y']}\n"
            f"Z: {info['dimension_z']}\n"
        )
    
    def _actualizar_visualizacion(self):
        if self.mesh is None:
            return

        self.plotter.clear()
        self.plotter.set_background('#FFFFFF')
        self.plotter.show_grid()

        intensidad = self.slider_intensidad.value() / 100.0
        suavizado = self.check_suavizado.isChecked()

        posicion_map = {
            0: (0, 0, 1),
            1: (0, 1, 0),
            2: (1, 0, 0),
        }
        posicion_luz = posicion_map[self.combo_posicion.currentIndex()]

        if self.modo_visualizacion == 'malla':
            pv_mesh = pv.wrap(self.mesh)
            self.plotter.add_mesh(
                pv_mesh,
                color='#DEE2E6',
                show_edges=False,
                smooth_shading=suavizado,
                ambient=intensidad * 0.3,
                diffuse=intensidad,
                specular=intensidad * 0.5,
            )
            self.plotter.remove_all_lights()
            self.plotter.add_light(pv.Light(
                position=posicion_luz,
                intensity=intensidad
            ))
        elif self.modo_visualizacion == 'nube':
            vertices, _ = extract_point_cloud(self.mesh)
            point_cloud = pv.PolyData(vertices)
            self.plotter.add_mesh(
                point_cloud,
                color='#718096',
                point_size=3,
                render_points_as_spheres=True
            )
        elif self.modo_visualizacion == 'proyeccion':
            self._visualizar_proyeccion()
            return

        self.plotter.reset_camera()

    def _actualizar_label_intensidad(self, valor):
        self.label_intensidad.setText(f'Intensidad: {valor}%')

    def _actualizar_label_focal(self, valor):
        self.focal_length = valor
        self.label_focal.setText(f'Distancia focal: {valor}px')

    def _actualizar_label_tz(self, valor):
        self.traduccion[2] = valor
        self.label_tz.setText(f'Distancia al objeto: {valor}')

    def _actualizar_label_rx(self, valor):
        self.rotacion[0] = valor
        self.label_rx.setText(f'Rotacion X: {valor}°')

    def _actualizar_label_ry(self, valor):
        self.rotacion[1] = valor
        self.label_ry.setText(f'Rotacion Y: {valor}°')

    def _visualizar_proyeccion(self):
        if self.mesh is None:
            return
        
        cx = self.resolucion[0] / 2
        cy = self.resolucion[1] / 2

        K = build_intrinsic_matrix(self.focal_length, cx, cy)
        Rt = build_extrinsic_matrix(self.traduccion, self.rotacion)

        vertices = np.array(self.mesh.vertices)
        puntos_2d, valid_mask = project_points(vertices, K, Rt)

        puntos_validos = puntos_2d[valid_mask]

        self.plotter.clear()
        self.plotter.set_background('#FFFFFF')

        ancho, alto = self.resolucion
        puntos_validos[:, 0] = np.clip(puntos_validos[:, 0], 0, ancho)
        puntos_validos[:, 1] = np.clip(puntos_validos[:, 1], 0, alto)

        puntos_3d = np.column_stack([
            puntos_validos[:, 0],
            puntos_validos[:, 1],
            np.zeros(len(puntos_validos))
        ])

        point_cloud = pv.PolyData(puntos_3d)
        self.plotter.add_mesh(
            point_cloud,
            color='#2B6CB0',
            point_size=2,
            render_points_as_spheres=True
        )

        self.plotter.view_xy()
        self.plotter.reset_camera()