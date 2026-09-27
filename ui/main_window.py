import sys
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QFrame, QComboBox, QSlider, QCheckBox,
    QLineEdit, QGroupBox, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
import pyvista as pv
from pyvistaqt import QtInteractor
from core.mesh_loader import load_mesh, extract_point_cloud
from core.camera import (
    build_intrinsic_matrix, build_camera_extrinsic, build_projector_extrinsic, project_points, get_camera_center, compute_fov
)
from core.geometry import condition_mesh
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

        # Parametros fisicos camara
        self.cam_focal_mm = 12.0
        self.cam_pixel_um = 4.65
        self.cam_width = 1024
        self.cam_height = 768
        self.cam_distance_mm = 1000.0

        # Parametros fisicos projector
        self.proj_focal_mm = 12.0
        self.proj_pixel_um = 4.50
        self.proj_width = 1024
        self.proj_height = 768
        self.proj_distance_mm = 1000.0
        self.proj_angle_deg = 12.0

        self.obj_alpha_deg = 0.0
        self.obj_beta_deg = 0.0
        
        self._build_ui()

    def _build_ui(self):
        widget_central = QWidget()
        self.setCentralWidget(widget_central)

        layout_principal = QHBoxLayout(widget_central)
        layout_principal.setSpacing(10)
        layout_principal.setContentsMargins(10, 10, 10, 10)

        # Panel izquierdo con Scroll
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(280)

        panel_scroll = QWidget()
        layout_izquierdo = QVBoxLayout(panel_scroll)
        layout_izquierdo.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout_izquierdo.setSpacing(12)
        layout_izquierdo.setContentsMargins(12, 12, 12, 12)

        scroll_area.setWidget(panel_scroll)

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

        # Grupo camara
        grupo_camara = QGroupBox('Cámara')
        layout_camara = QVBoxLayout(grupo_camara)
        layout_camara.setSpacing(6)

        self.campo_cam_focal = self._crear_campo(layout_camara, 'Longitud focal (mm):', '12.0')
        self.campo_cam_pixel = self._crear_campo(layout_camara, 'Paso de píxel (µm):', '4.65')
        self.campo_cam_width, self.campo_cam_height = self._crear_campo_resolucion(layout_camara, 'Resolución:', '1024', '768')
        self.campo_cam_distance = self._crear_campo(layout_camara, 'Distancia (mm):', '1000.0')
        
        self.campo_obj_alpha = self._crear_campo(layout_camara, 'Rotación objeto X (°)', '0.0')
        self.campo_obj_beta = self._crear_campo(layout_camara, 'Rotación objeto Y (°)', '0.0')

        # Grupo proyector
        grupo_proyector = QGroupBox('Proyector')
        layout_proyector = QVBoxLayout(grupo_proyector)
        layout_proyector.setSpacing(6)

        self.campo_proj_focal = self._crear_campo(layout_proyector, 'Longitud focal (mm):', '12.0')
        self.campo_proj_pixel = self._crear_campo(layout_proyector, 'Paso de píxel (µm):', '4.50')
        self.campo_proj_width, self.campo_proj_height = self._crear_campo_resolucion(layout_proyector, 'Resolución:', '1024', '768')
        self.campo_proj_distance = self._crear_campo(layout_proyector, 'Distancia (mm):', '1000.0')
        self.campo_proj_angle = self._crear_campo(layout_proyector, 'Ángulo (°):', '12.0')

        # Boton para aplicar configuracion
        self.btn_aplicar = QPushButton('Aplicar configuración')
        self.btn_aplicar.setFixedHeight(40)
        self.btn_aplicar.clicked.connect(self._aplicar_configuracion)

        # Separador
        sep4 = QFrame()
        sep4.setFrameShape(QFrame.Shape.HLine)
        sep4.setFixedHeight(1)

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
        layout_izquierdo.addWidget(grupo_camara)
        layout_izquierdo.addWidget(grupo_proyector)
        layout_izquierdo.addWidget(self.btn_aplicar)
        layout_izquierdo.addWidget(sep4)
        layout_izquierdo.addWidget(titulo_info)
        layout_izquierdo.addWidget(self.label_nombre_titulo)
        layout_izquierdo.addWidget(self.label_nombre)
        layout_izquierdo.addWidget(self.label_puntos_titulo)
        layout_izquierdo.addWidget(self.label_puntos)
        layout_izquierdo.addWidget(self.label_caras_titulo)
        layout_izquierdo.addWidget(self.label_caras)
        layout_izquierdo.addWidget(self.label_dim_titulo)
        layout_izquierdo.addWidget(self.label_dim)

        # Panel derecho - Visualizacion 3D
        self.plotter = QtInteractor(self)

        layout_principal.addWidget(scroll_area)
        layout_principal.addWidget(self.plotter)

    def _crear_campo(self, layout, etiqueta, valor_default):
        label = QLabel(etiqueta)
        label.setObjectName('label_valor')
        campo = QLineEdit(valor_default)
        campo.setValidator(QDoubleValidator())
        layout.addWidget(label)
        layout.addWidget(campo)
        return campo

    def _crear_campo_resolucion(self, layout, etiqueta, valor_w, valor_h):
        label = QLabel(etiqueta)
        label.setObjectName('label_valor')
        layout.addWidget(label)
        fila = QWidget()
        fila_layout = QHBoxLayout(fila)
        fila_layout.setContentsMargins(0, 0, 0, 0)
        fila_layout.setSpacing(6)
        campo_w = QLineEdit(valor_w)
        campo_w.setValidator(QDoubleValidator())
        label_x = QLabel('x')
        label_x.setObjectName('label_valor')
        campo_h = QLineEdit(valor_h)
        campo_h.setValidator(QDoubleValidator())
        fila_layout.addWidget(campo_w)
        fila_layout.addWidget(label_x)
        fila_layout.addWidget(campo_h)
        layout.addWidget(fila)
        return campo_w, campo_h
    
    def _aplicar_configuracion(self):
        try:
            self.cam_focal_mm = float(self.campo_cam_focal.text())
            self.cam_pixel_um = float(self.campo_cam_pixel.text())
            self.cam_width = int(float(self.campo_cam_width.text()))
            self.cam_height = int(float(self.campo_cam_height.text()))
            self.cam_distance_mm = float(self.campo_cam_distance.text())

            self.proj_focal_mm = float(self.campo_proj_focal.text())
            self.proj_pixel_um = float(self.campo_proj_pixel.text())
            self.proj_width = int(float(self.campo_proj_width.text()))
            self.proj_height = int(float(self.campo_proj_height.text()))
            self.proj_distance_mm = float(self.campo_proj_distance.text())
            self.proj_angle_deg = float(self.campo_proj_angle.text())

            self.obj_alpha_deg = float(self.campo_obj_alpha.text())
            self.obj_beta_deg = float(self.campo_obj_beta.text())

            if self.mesh is not None:
                self._actualizar_visualizacion()
        
        except ValueError:
            QMessageBox.critical(self, 'Error', 'Por favor verifica que todos los campos tengan valores numéricos válidos.')

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
        
    def _actualizar_label_intensidad(self, valor):
        self.label_intensidad.setText(f'Intensidad: {valor}%')

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

    def _visualizar_proyeccion(self):
        if self.mesh is None:
            return
        
        K, f_px, cx, cy = build_intrinsic_matrix(self.cam_focal_mm, self.cam_pixel_um, self.cam_width, self.cam_height)

        vertices = np.array(self.mesh.vertices)

        vertices_cam, scale, Lx, Ly = condition_mesh(vertices, self.obj_alpha_deg, self.obj_beta_deg, f_px,
        self.cam_width, self.cam_height, self.cam_distance_mm)

        Rt_identity = np.hstack([np.eye(3), np.zeros((3, 1))])
        puntos_2d, valid_mask = project_points(vertices_cam, K, Rt_identity)
        puntos_validos = puntos_2d[valid_mask]

        if len(puntos_validos) == 0:
            QMessageBox.warning(self, 'Aviso', 'No hay puntos visibles en la configuración actual.')
            return

        puntos_validos[:, 0] = np.clip(puntos_validos[:, 0], 0, self.cam_width)
        puntos_validos[:, 1] = np.clip(puntos_validos[:, 1], 0, self.cam_height)

        puntos_3d = np.column_stack([
            puntos_validos[:, 0],
            puntos_validos[:, 1],
            np.zeros(len(puntos_validos))
        ])

        self.plotter.clear()
        self.plotter.set_background('#FFFFFF')

        point_cloud = pv.PolyData(puntos_3d)
        self.plotter.add_mesh(
            point_cloud,
            color='#2B6CB0',
            point_size=2,
            render_points_as_spheres=True
        )

        self.plotter.view_xy()
        self.plotter.reset_camera()