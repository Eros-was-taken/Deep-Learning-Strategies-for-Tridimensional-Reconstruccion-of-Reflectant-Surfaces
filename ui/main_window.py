import sys
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QFrame, QComboBox, QSlider, QCheckBox,
    QLineEdit, QGroupBox, QScrollArea, QTabWidget, QStackedWidget, QApplication
)
from PySide6.QtCore import Qt, QLocale
from PySide6.QtGui import QDoubleValidator
import pyvista as pv
from pyvistaqt import QtInteractor
from core.mesh_loader import load_mesh, extract_point_cloud
from core.camera import (
    build_intrinsic_matrix, build_camera_extrinsic, build_projector_extrinsic, project_points, get_camera_center, compute_fov
)
from core.geometry import condition_mesh, compute_system_geometry
from core.visibility import rasterize_zbuffer, compute_point_map_and_normals, camera_to_projector_coords, compute_projector_visibility, ETQ_ILUMINADO
from core.radiometry import (ideal_fringe_image, NIVELES_ESPECULARES, specular_map, expand_specular, degrade_fringe_image, saturated_pixels)
from core.phase import geometric_phase, psa_phase, phase_coverage
from ui.image_viewer import ImageViewer
from ui.styles import STYLE

class ValidadorNumerico(QDoubleValidator):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLocale(QLocale(QLocale.Language.C))
        self.setNotation(QDoubleValidator.Notation.ScientificNotation)

    def validate(self, texto, pos):
        return super().validate(texto.replace(',', '.'), pos)

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

        # Parametros fisicos sisara
        self.sis_focal_mm = 12.0
        self.sis_pixel_um = 4.65
        self.sis_width = 1024
        self.sis_height = 768
        self.sis_distance_mm = 1000.0

        # Parametros fisicos projector
        self.proj_focal_mm = 12.0
        self.proj_pixel_um = 4.50
        self.proj_width = 1024
        self.proj_height = 768
        self.proj_distance_mm = 1000.0
        self.proj_angle_deg = 12.0

        self.obj_alpha_deg = 0.0
        self.obj_beta_deg = 0.0

        self._cache_visibilidad = None

        # Parametros de las franjas
        self.fr_periodo_px = 64.0
        self.fr_u0_px = 512.5
        self.fr_k = 1
        self.fr_n_pasos = 12
        self.fr_ambiente = 0.05

        self.fr_sigma = 0.01
        self.fr_semilla = 2026
        self.fr_gamma = 2.6
        self.fr_etapa = 0

        self.fr_kappa = 60.0
        self.fr_g = 1.10
        self.fr_o = 0.30
        self.fr_sigma_esp = 9.0
        
        self._build_ui()

    def _build_ui(self):
        widget_central = QWidget()
        self.setCentralWidget(widget_central)

        layout_principal = QHBoxLayout(widget_central)
        layout_principal.setSpacing(10)
        layout_principal.setContentsMargins(10, 10, 10, 10)

        # Panel izquierdo con pestañas
        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(280)

        # Pestaña 1 - Visualizacion
        tab_vis = QWidget()
        layout_vis = QVBoxLayout(tab_vis)
        layout_vis.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout_vis.setSpacing(12)
        layout_vis.setContentsMargins(12, 12, 12, 12)
        
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
        self.combo_modo.addItems(['Malla triangular', 'Nube de puntos', 'Proyeccion camara', 'Sistema completo', 'Mapa de visibilidad', 'Mapa de normales',
                                'Mapa de coordenadas del proyector', 'Mapa de sombras (Ω_map)', 'Imagen de franjas'])
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

        layout_vis.addWidget(self.btn_cargar)
        layout_vis.addWidget(sep1)
        layout_vis.addWidget(titulo_modo)
        layout_vis.addWidget(self.combo_modo)
        layout_vis.addWidget(sep2)
        layout_vis.addWidget(titulo_luz)
        layout_vis.addWidget(self.label_intensidad)
        layout_vis.addWidget(self.slider_intensidad)
        layout_vis.addWidget(titulo_posicion)
        layout_vis.addWidget(self.combo_posicion)
        layout_vis.addWidget(self.check_suavizado)
        layout_vis.addWidget(sep3)
        layout_vis.addWidget(titulo_info)
        layout_vis.addWidget(self.label_nombre_titulo)
        layout_vis.addWidget(self.label_nombre)
        layout_vis.addWidget(self.label_puntos_titulo)
        layout_vis.addWidget(self.label_puntos)
        layout_vis.addWidget(self.label_caras_titulo)
        layout_vis.addWidget(self.label_caras)
        layout_vis.addWidget(self.label_dim_titulo)
        layout_vis.addWidget(self.label_dim)

        # Pestaña 2 - Sistema
        tab_sistema = QWidget()
        scroll_sis = QScrollArea()
        scroll_sis.setWidgetResizable(True)
        panel_sis = QWidget()
        layout_sis = QVBoxLayout(panel_sis)
        layout_sis.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout_sis.setSpacing(8)
        layout_sis.setContentsMargins(12, 12, 12, 12)
        scroll_sis.setWidget(panel_sis)

        layout_sis_outer = QVBoxLayout(tab_sistema)
        layout_sis_outer.setContentsMargins(0, 0, 0, 0)
        layout_sis_outer.addWidget(scroll_sis)

        # Grupo camara
        grupo_camara = QGroupBox('Cámara')
        layout_camara = QVBoxLayout(grupo_camara)
        layout_camara.setSpacing(6)

        self.campo_cam_focal = self._crear_campo(layout_camara, 'Longitud focal (mm):', '12.0')
        self.campo_cam_pixel = self._crear_campo(layout_camara, 'Paso de píxel (µm):', '4.65')
        self.campo_cam_width, self.campo_cam_height = self._crear_campo_resolucion(layout_camara, 'Resolución:', '1024', '768')
        self.campo_cam_distance = self._crear_campo(layout_camara, 'Distancia (mm):', '1000.0')

        # Grupo proyector
        grupo_proyector = QGroupBox('Proyector')
        layout_proyector = QVBoxLayout(grupo_proyector)
        layout_proyector.setSpacing(6)

        self.campo_proj_focal = self._crear_campo(layout_proyector, 'Longitud focal (mm):', '12.0')
        self.campo_proj_pixel = self._crear_campo(layout_proyector, 'Paso de píxel (µm):', '4.50')
        self.campo_proj_width, self.campo_proj_height = self._crear_campo_resolucion(layout_proyector, 'Resolución:', '1024', '768')
        self.campo_proj_distance = self._crear_campo(layout_proyector, 'Distancia (mm):', '1000.0')
        self.campo_proj_angle = self._crear_campo(layout_proyector, 'Ángulo (°):', '12.0')

        # Grupo orientacion del objeto
        grupo_objeto = QGroupBox('Orientación del objeto')
        layout_objeto = QVBoxLayout(grupo_objeto)
        layout_objeto.setSpacing(6)
        
        self.campo_obj_alpha = self._crear_campo(layout_objeto, 'Rotación objeto X (°)', '0.0')
        self.campo_obj_beta = self._crear_campo(layout_objeto, 'Rotación objeto Y (°)', '0.0')

        # Boton para aplicar configuracion
        btn_aplicar = QPushButton('Aplicar configuración')
        btn_aplicar.setFixedHeight(40)
        btn_aplicar.clicked.connect(self._aplicar_configuracion)

        titulo_modo_sis = QLabel('MODO DE VISUALIZACIÓN')
        titulo_modo_sis.setObjectName('label_titulo')

        self.combo_modo_sis = QComboBox()
        self.combo_modo_sis.addItems(['Malla triangular', 'Nube de puntos', 'Proyección cámara', 'Sistema completo', 'Mapa de visibilidad', 'Mapa de normales',
                                    'Mapa de coordenadas del proyector', 'Mapa de sombras (Ω_map)', 'Imagen de franjas'])
        self.combo_modo_sis.setCurrentIndex(0)
        self.combo_modo_sis.currentIndexChanged.connect(self._cambiar_modo_sis)

        sep_sis = QFrame()
        sep_sis.setFrameShape(QFrame.Shape.HLine)
        sep_sis.setFixedHeight(1)

        layout_sis.addWidget(titulo_modo_sis)
        layout_sis.addWidget(self.combo_modo_sis)
        layout_sis.addWidget(sep_sis)

        layout_sis.addWidget(grupo_camara)
        layout_sis.addWidget(grupo_proyector)
        layout_sis.addWidget(grupo_objeto)
        layout_sis.addWidget(btn_aplicar)

        # Pestaña 3 - Franjas
        tab_franjas = QWidget()
        scroll_fr = QScrollArea()
        scroll_fr.setWidgetResizable(True)
        panel_fr = QWidget()
        layout_fr = QVBoxLayout(panel_fr)
        layout_fr.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout_fr.setSpacing(8)
        layout_fr.setContentsMargins(12, 12, 12, 12)
        scroll_fr.setWidget(panel_fr)

        layout_fr_outer = QVBoxLayout(tab_franjas)
        layout_fr_outer.setContentsMargins(0, 0, 0, 0)
        layout_fr_outer.addWidget(scroll_fr)

        # Grupo imagen mostrada (controles en vivo)
        grupo_imagen = QGroupBox('Imagen mostrada')
        layout_imagen = QVBoxLayout(grupo_imagen)
        layout_imagen.setSpacing(6)

        label_etapa = QLabel('Etapa')
        label_etapa.setObjectName('label_valor')
        self.combo_fr_etapa = QComboBox()
        self.combo_fr_etapa.addItems(['Ideal', 'Con ruido', 'Con ruido y gamma', 'Con especular (sin recortar)', 'Final (recortada)', 'Máscara de saturación', 'Fase continua Φ_geom',
                                    'Fase envuelta φ_ideal (PSA)'])
        self.combo_fr_etapa.currentIndexChanged.connect(self._cambiar_etapa_franja)

        self.label_fr_k = QLabel('Imagen k: 1 de 12')
        self.label_fr_k.setObjectName('label_valor')
        self.slider_fr_k = QSlider(Qt.Orientation.Horizontal)
        self.slider_fr_k.setMinimum(1)
        self.slider_fr_k.setMaximum(12)
        self.slider_fr_k.setValue(1)
        self.slider_fr_k.valueChanged.connect(self._cambiar_k_franja)

        layout_imagen.addWidget(label_etapa)
        layout_imagen.addWidget(self.combo_fr_etapa)
        layout_imagen.addWidget(self.label_fr_k)
        layout_imagen.addWidget(self.slider_fr_k)

        # Grupo patron de franjas
        grupo_patron = QGroupBox('Patrón de franjas')
        layout_patron = QVBoxLayout(grupo_patron)
        layout_patron.setSpacing(6)

        self.campo_fr_periodo = self._crear_campo(layout_patron, 'Periodo Tp (px del proyector):', '64.0')
        self.campo_fr_pasos = self._crear_campo(layout_patron, 'Número de pasos N:', '12')
        self.campo_fr_u0 = self._crear_campo(layout_patron, 'Origen de fase u0p (px):', '512.5')
        self.campo_fr_ambiente = self._crear_campo(layout_patron, 'Intensidad ambiente I_amb:', '0.05')

        # Grupo degradaciones
        grupo_degr = QGroupBox('Degradaciones')
        layout_degr = QVBoxLayout(grupo_degr)
        layout_degr.setSpacing(6)

        self.campo_fr_sigma = self._crear_campo(layout_degr, 'Ruido: desviación σ:', '0.01')
        self.campo_fr_semilla = self._crear_campo(layout_degr, 'Ruido: semilla:', '2026')
        self.campo_fr_gamma = self._crear_campo(layout_degr, 'Gamma γ:', '2.6')

        # Grupo reflexion especular
        grupo_esp = QGroupBox('Reflexión especular')
        layout_esp = QVBoxLayout(grupo_esp)
        layout_esp.setSpacing(6)

        label_nivel = QLabel('Nivel preestablecido:')
        label_nivel.setObjectName('label_valor')
        self.combo_fr_nivel = QComboBox()
        self.combo_fr_nivel.addItems(list(NIVELES_ESPECULARES.keys()))
        self.combo_fr_nivel.setCurrentIndex(2)
        layout_esp.addWidget(label_nivel)
        layout_esp.addWidget(self.combo_fr_nivel)

        self.campo_fr_kappa = self._crear_campo(layout_esp, 'Exponente especular κs:', '60')
        self.campo_fr_g = self._crear_campo(layout_esp, 'Amplificación g:', '1.10')
        self.campo_fr_o =  self._crear_campo(layout_esp, 'Contribución aditiva o:', '0.30')
        self.campo_fr_sigma_esp =  self._crear_campo(layout_esp, 'Extensión del destello σℓ (px):', '9.0')
        self.combo_fr_nivel.currentIndexChanged.connect(self._cargar_nivel_especular)

        btn_aplicar_fr = QPushButton('Aplicar y mostrar franjas')
        btn_aplicar_fr.setFixedHeight(40)
        btn_aplicar_fr.clicked.connect(self._aplicar_franjas)

        layout_fr.addWidget(grupo_imagen)
        layout_fr.addWidget(grupo_patron)
        layout_fr.addWidget(grupo_degr)
        layout_fr.addWidget(grupo_esp)
        layout_fr.addWidget(btn_aplicar_fr)

        # Agregar pestañas
        self.tabs.addTab(tab_vis, 'Visualización')
        self.tabs.addTab(tab_sistema, 'Sistema')
        self.tabs.addTab(tab_franjas, 'Franjas')

        # Panel derecho - Visualizacion 3D
        self.plotter = QtInteractor(self)
        self.visor_2d = ImageViewer()

        self.stack = QStackedWidget()
        self.stack.addWidget(self.plotter)
        self.stack.addWidget(self.visor_2d)

        layout_principal.addWidget(self.tabs)
        layout_principal.addWidget(self.stack)

    def _crear_campo(self, layout, etiqueta, valor_default):
        label = QLabel(etiqueta)
        label.setObjectName('label_valor')
        campo = QLineEdit(valor_default)
        campo.setValidator(ValidadorNumerico())
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
        campo_w.setValidator(ValidadorNumerico())
        label_x = QLabel('x')
        label_x.setObjectName('label_valor')
        campo_h = QLineEdit(valor_h)
        campo_h.setValidator(ValidadorNumerico())
        fila_layout.addWidget(campo_w)
        fila_layout.addWidget(label_x)
        fila_layout.addWidget(campo_h)
        layout.addWidget(fila)
        return campo_w, campo_h
    
    def _aplicar_configuracion(self):
        try:
            self.sis_focal_mm = float(self.campo_cam_focal.text())
            self.sis_pixel_um = float(self.campo_cam_pixel.text())
            self.sis_width = int(float(self.campo_cam_width.text()))
            self.sis_height = int(float(self.campo_cam_height.text()))
            self.sis_distance_mm = float(self.campo_cam_distance.text())

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
            self._cache_visibilidad = None
            self._actualizar_info()
            self._actualizar_visualizacion()

        except RuntimeError as e:
            QMessageBox.critical(self, 'Error', str(e))
    
    def _cambiar_modo(self):
        indice = self.combo_modo.currentIndex()
        modos = ['malla', 'nube', 'proyeccion', 'sistema', 'visibilidad', 'normales', 'proyector', 'sombras', 'franjas']
        self.modo_visualizacion = modos[indice]

        if indice >= 2 and self.tabs.currentIndex() == 0:
            self.tabs.setCurrentIndex(1)
        
        self.combo_modo_sis.blockSignals(True)
        self.combo_modo_sis.setCurrentIndex(indice)
        self.combo_modo_sis.blockSignals(False)

        self._actualizar_visualizacion()
        
    def _cambiar_modo_sis(self):
        index = self.combo_modo_sis.currentIndex()
        self.combo_modo.setCurrentIndex(index)
        if index in [0, 1]:
            self.tabs.setCurrentIndex(0)

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
        if self.modo_visualizacion in ('visibilidad', 'normales', 'proyector', 'sombras', 'franjas'):
            self.stack.setCurrentIndex(1)
        else:
            self.stack.setCurrentIndex(0)

        if self.mesh is None:
            return
        
        if self.modo_visualizacion in ('visibilidad', 'normales', 'proyector', 'sombras', 'franjas'):
            self._visualizar_visibilidad()
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
        elif self.modo_visualizacion == 'sistema':
            self._visualizar_sistema()
            return

        self.plotter.reset_camera()

    def _visualizar_proyeccion(self):
        if self.mesh is None:
            return
        
        K, f_px, cx, cy = build_intrinsic_matrix(self.sis_focal_mm, self.sis_pixel_um, self.sis_width, self.sis_height)

        vertices = np.array(self.mesh.vertices)

        vertices_cam, scale, Lx, Ly = condition_mesh(vertices, self.obj_alpha_deg, self.obj_beta_deg, f_px,
        self.sis_width, self.sis_height, self.sis_distance_mm)

        Rt_identity = np.hstack([np.eye(3), np.zeros((3, 1))])
        puntos_2d, valid_mask = project_points(vertices_cam, K, Rt_identity)
        puntos_validos = puntos_2d[valid_mask]

        if len(puntos_validos) == 0:
            QMessageBox.warning(self, 'Aviso', 'No hay puntos visibles en la configuración actual.')
            return

        puntos_validos[:, 0] = np.clip(puntos_validos[:, 0], 0, self.sis_width)
        puntos_validos[:, 1] = np.clip(puntos_validos[:, 1], 0, self.sis_height)

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
    
    def _visualizar_sistema(self):
        if self.mesh is None:
            return
        
        K, f_px, cx, cy = build_intrinsic_matrix(self.sis_focal_mm, self.sis_pixel_um, self.sis_width, self.sis_height)

        vertices = np.array(self.mesh.vertices)

        vertices_cam, scale, Lx, Ly = condition_mesh(vertices, self.obj_alpha_deg, self.obj_beta_deg, f_px,
        self.sis_width, self.sis_height, self.sis_distance_mm)

        cam_center, proj_center, origin = compute_system_geometry(self.sis_distance_mm, self.proj_distance_mm, self.proj_angle_deg,
        scale, self.sis_distance_mm)

        self.plotter.clear()
        self.plotter.set_background('#FFFFFF')

        R_cam_inv = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], dtype=float)

        vertices_obj = (R_cam_inv @ (vertices_cam - [0, 0, self.sis_distance_mm]).T).T
        pv_mesh = pv.PolyData(vertices_obj)
        if hasattr(self.mesh, 'faces') and len(self.mesh.faces) > 0:
            pv_mesh = pv.wrap(self.mesh)
            pv_mesh.points = vertices_obj

        self.plotter.add_mesh(pv_mesh, color='#DEE2E6', show_edges=False, smooth_shading=True, opacity=0.8)

        cam_point = pv.PolyData(cam_center.reshape(1, 3))
        self.plotter.add_mesh(cam_point, color='#2B6CB0', point_size=20, render_points_as_spheres=True)
        self.plotter.add_point_labels(cam_center.reshape(1, 3), ['Cámara'], font_size=16, text_color='#2B6CB0', bold=True,
        show_points=False, always_visible=True)

        proj_point = pv.PolyData(proj_center.reshape(1, 3))
        self.plotter.add_mesh(proj_point, color='#E53E3E', point_size=20, render_points_as_spheres=True)
        self.plotter.add_point_labels(proj_center.reshape(1, 3), ['Proyector'], font_size=16, text_color='#E53E3E', bold=True,
        show_points=False, always_visible=True)

        linea_cam = pv.Line(cam_center, origin)
        self.plotter.add_mesh(linea_cam, color='#2B6CB0', line_width=2)

        linea_proj = pv.Line(proj_center, origin)
        self.plotter.add_mesh(linea_proj, color='#E53E3E', line_width=2)

        linea_base = pv.Line(cam_center, proj_center)
        self.plotter.add_mesh(linea_base, color='718096', line_width=3)

        self.plotter.show_grid()
        self.plotter.view_xy()
        self.plotter.camera.elevation = 15
        self.plotter.reset_camera()
    
    def _visualizar_visibilidad(self):
        clave = (self.sis_focal_mm, self.sis_pixel_um, self.sis_width, self.sis_height, self.sis_distance_mm, self.obj_alpha_deg, self.obj_beta_deg)

        cache = self._cache_visibilidad
        if cache is None or cache['clave'] != clave:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                K, f_px, cx, cy = build_intrinsic_matrix(self.sis_focal_mm, self.sis_pixel_um, self.sis_width, self.sis_height)

                vertices = np.array(self.mesh.vertices)
                vertices_cam, scale, Lx, Ly = condition_mesh(vertices, self.obj_alpha_deg, self.obj_beta_deg, f_px, self.sis_width, self.sis_height, self.sis_distance_mm)

                faces = np.array(self.mesh.faces)
                depth, face_id, omega_cam = rasterize_zbuffer(vertices_cam, faces, K, self.sis_width, self.sis_height)

                P, N = compute_point_map_and_normals(vertices_cam, faces, face_id, depth, K)
            finally:
                QApplication.restoreOverrideCursor()
            
            cache = {'clave': clave, 'depth': depth, 'face_id': face_id, 'omega_cam': omega_cam, 'P': P, 'N': N, 'vertices_cam': vertices_cam, 'faces': faces}
            self._cache_visibilidad = cache
        
        omega = cache['omega_cam']
        porcentaje = 100.0 * omega.sum() / omega.size
    
        if self.modo_visualizacion == 'proyector':
            K_p, _, _, _ = build_intrinsic_matrix(self.proj_focal_mm, self.proj_pixel_um, self.proj_width, self.proj_height)
            _, R_po, t_po = build_projector_extrinsic(self.proj_distance_mm, self.proj_angle_deg)
            u_p, v_p, z_p = camera_to_projector_coords(cache['P'], R_po, t_po, K_p, self.sis_distance_mm)
            dentro = (omega & (u_p >= 1) & (u_p <= self.proj_width) & (v_p >= 1) & (v_p <= self.proj_height))
            self.visor_2d.show_depth(u_p, omega, title=f'u_p · {dentro.sum():,} de {omega.sum():,} píxeles caen dentro del proyector', cbar_label='Columna del proyector u_p (px)')
            return
        
        if self.modo_visualizacion == 'sombras':
            etiquetas, _ = self._asegurar_visibilidad_proyector(cache, omega)
            n_map = int((etiquetas == ETQ_ILUMINADO).sum())
            self.visor_2d.show_categories(etiquetas, ['Fuera de Ω_cam', 'Iluminado (Ω_map)', 'Fuera del campo del proyector', 'Sombra propia', 'Sombra proyectada'],
                                        ['#E2E8F0', '#F4D35E', '#9AA5B1', '#3B4A6B', '#1B1F3B'], title=f'Ω_map: {n_map:,} de {omega.sum():,} píxeles de Ω_cam iluminados')
            return
        
        if self.modo_visualizacion == 'franjas':
            self._visualizar_franjas(cache, omega)
            return

        if self.modo_visualizacion == 'normales':
            self.visor_2d.show_normals(cache['N'], omega, title=f'Normales (RGB = (n+1)/2 · {omega.sum():,} píxeles)')
        else:
            self.visor_2d.show_depth(cache['depth'], omega, title=f'Ω_cam: {omega.sum():,} píxeles ({porcentaje:.1f} %)')
    
    def _asegurar_visibilidad_proyector(self, cache, omega):
        clave_proy = (self.proj_focal_mm, self.proj_pixel_um, self.proj_width, self.proj_height, self.proj_distance_mm, self.proj_angle_deg)
        if cache.get('clave_proy') != clave_proy:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                K_p, _, _, _ = build_intrinsic_matrix(self.proj_focal_mm, self.proj_pixel_um, self.proj_width, self.proj_height)
                _, R_po, t_po = build_projector_extrinsic(self.proj_distance_mm, self.proj_angle_deg)
                etiquetas, info = compute_projector_visibility(cache['vertices_cam'], cache['faces'], cache['P'], cache['N'], omega, R_po, t_po, K_p, self.sis_distance_mm,
                                                            self.proj_width, self.proj_height)
            finally:
                QApplication.restoreOverrideCursor()
            cache['clave_proy'] = clave_proy
            cache['etiquetas'] = etiquetas
            cache['info_proy'] = info
        return cache['etiquetas'], cache['info_proy']
    
    def _aplicar_franjas(self):
        try:
            periodo = float(self.campo_fr_periodo.text())
            n_pasos = int(float(self.campo_fr_pasos.text()))
            u0 = float(self.campo_fr_u0.text())
            ambiente = float(self.campo_fr_ambiente.text())
            sigma = float(self.campo_fr_sigma.text())
            semilla = int(float(self.campo_fr_semilla.text()))
            gamma = float(self.campo_fr_gamma.text())
            kappa = float(self.campo_fr_kappa.text())
            g = float(self.campo_fr_g.text())
            o = float(self.campo_fr_o.text())
            sigma_esp = float(self.campo_fr_sigma_esp.text())
        except ValueError:
            QMessageBox.critical(self, 'Error', 'Por favor verifica que todos los campos tengan valores numéricos válidos.')
            return

        if periodo <= 0:
            QMessageBox.critical(self, 'Error', 'El periodo Tp debe de ser mayor que cero.')
            return
        if n_pasos < 1:
            QMessageBox.critical(self, 'Error', 'El número de pasos N debe ser al menos 1.')
            return
        if not 0.0 <= ambiente <= 1.0:
            QMessageBox.critical(self, 'Error', 'La intensidad ambiente debe estar entre 0 y 1.')
            return
        if sigma < 0:
            QMessageBox.critical(self, 'Error', 'La desviación del ruido σ no puede ser negativa.')
            return
        if not 0 <= semilla < 2 ** 32:
            QMessageBox.critical(self, 'Error', 'La semilla debe ser un entero entre 0 y 4294967295.')
            return
        if gamma <= 0:
            QMessageBox.critical(self, 'Error', 'Gamma γ debe ser mayor que cero.')
            return
        if kappa <= 0:
            QMessageBox.critical(self, 'Error', 'El exponente especular κs debe ser mayor que cero.')
            return
        if g < 0 or o < 0:
            QMessageBox.critical(self, 'Error', 'g y o no pueden ser negativos.')
            return
        if sigma_esp < 0:
            QMessageBox.critical(self, 'Error', 'La extensión del destello σℓ no puede ser negativa.')
            return
        
        self.fr_periodo_px = periodo
        self.fr_n_pasos = n_pasos
        self.fr_u0_px = u0
        self.fr_ambiente = ambiente
        self.fr_sigma = sigma
        self.fr_semilla = semilla
        self.fr_gamma = gamma
        self.fr_kappa = kappa
        self.fr_g = g
        self.fr_o = o
        self.fr_sigma_esp = sigma_esp

        # Selector de k debe cubrir las N imagenes
        self.slider_fr_k.blockSignals(True)
        self.slider_fr_k.setMaximum(n_pasos)
        self.fr_k = self.slider_fr_k.value()
        self.slider_fr_k.blockSignals(False)
        self.label_fr_k.setText(f'Imagen k: {self.fr_k} de {self.fr_n_pasos}')

        indice_franjas = self.combo_modo.findText('Imagen de franjas')
        if self.combo_modo.currentIndex() != indice_franjas:
            self.combo_modo.setCurrentIndex(indice_franjas)
        else:
            self._actualizar_visualizacion()
        
    def _cambiar_k_franja(self, valor):
        self.fr_k = valor
        self.label_fr_k.setText(f'Imagen k: {valor} de {self.fr_n_pasos}')
        if self.mesh is not None and self.modo_visualizacion == 'franjas':
            self._actualizar_visualizacion()
    
    def _cambiar_etapa_franja(self, indice):
        self.fr_etapa = indice
        if self.mesh is not None and self.modo_visualizacion == 'franjas':
            self._actualizar_visualizacion()

    def _obtener_especular(self, cache, info, omega_map):
        clave = (cache['clave_proy'], self.fr_kappa, self.fr_sigma_esp)
        if cache.get('clave_proy') != clave:
            S = specular_map(cache['P'], cache['N'], info['C_proy'], omega_map, self.fr_kappa)
            cache['S_exp'] = expand_specular(S, omega_map, self.fr_sigma_esp)
            cache['clave_espec'] = clave
        return cache['S_exp']

    def _etapas_franja(self, cache, info, omega, omega_map, k):
        S_exp = self._obtener_especular(cache, info, omega_map)
        ideal, _, _ = ideal_fringe_image(info['u_p'], info['n_dot_l'], omega, omega_map, period_px=self.fr_periodo_px, u0_px=self.fr_u0_px, k=self.fr_k, n_steps=self.fr_n_pasos,
                                        ambient=self.fr_ambiente)
        return degrade_fringe_image(ideal, omega_map, S_exp, sigma=self.fr_sigma, gamma=self.fr_gamma, seed=self.fr_semilla, k=k, g=self.fr_g, o=self.fr_o)

    def _visualizar_franjas(self, cache, omega):
        etiquetas, info = self._asegurar_visibilidad_proyector(cache, omega)
        omega_map = etiquetas == ETQ_ILUMINADO

        if self.fr_etapa == 5:
            self._mostrar_saturacion(cache, info, omega, omega_map)
            return

        if self.fr_etapa in (6, 7):
            self._mostrar_fase(info, omega, omega_map, envuelta=(self.fr_etapa == 7))
            return

        etapas = self._etapas_franja(cache, info, omega, omega_map, self.fr_k)
        claves = ['ideal', 'ruido', 'gamma', 'pre', 'final']
        nombres = ['ideal', 'con ruido', 'con ruido y gamma', 'con especular (sin recortar)', 'final (recortada)']
        imagen = etapas[claves[self.fr_etapa]]
        vmax = max(1.0, float(imagen.max()))

        titulo = (f'Imagen {nombres[self.fr_etapa]} I_k (k = {self.fr_k} de {self.fr_n_pasos}) · T_p = {self.fr_periodo_px:g} px')
        if vmax > 1.0:
            titulo += f' · máximo = {vmax:.2f}'
        self.visor_2d.show_intensity(imagen, title=titulo, vmax=vmax)

    def _mostrar_saturacion(self, cache, info, omega, omega_map):
        clave = (cache['clave_proy'], self.fr_periodo_px, self.fr_u0_px, self.fr_n_pasos, self.fr_ambiente, self.fr_sigma, self.fr_semilla, self.fr_gamma, self.fr_kappa, self.fr_g, self.fr_o,
                self.fr_sigma_esp)
        if cache.get('clave_sat') != clave:
            saturados = np.zeros(omega.shape, dtype=bool)
            for k in range(1, self.fr_n_pasos + 1):
                etapas = self._etapas_franja(cache, info, omega, omega_map, k)
                saturados |= saturated_pixels(etapas['pre'], omega_map)
            cache['mascara_sat'] = saturados
            cache['clave_sat'] = clave

        saturados = cache['mascara_sat']
        categorias = np.zeros(omega.shape, dtype=np.uint8)
        categorias[omega_map] = 1
        categorias[saturados] = 2
        n_sat = int(saturados.sum())
        n_map = max(int(omega_map.sum()), 1)
        self.visor_2d.show_categories(categorias, ['Fuera de Ω_map', 'Sin saturar', 'Saturado'], ['#E2E8F0', '#94A3B8', '#DC2626'],
                                    title=f'Máscara de saturación (unión de las {self.fr_n_pasos} imágenes): {n_sat:,} px ({100.0 * n_sat / n_map:.2f} % de Ω_map)')

    def _mostrar_fase(self, info, omega, omega_map, envuelta):
        fase_geom = geometric_phase(info['u_p'], omega_map, self.fr_periodo_px, self.fr_u0_px)
        r_ideal = phase_coverage(fase_geom, omega_map)

        if not envuelta:
            self.visor_2d.show_phase(fase_geom, omega_map, title=f'Fase continua Φ_geom · T_p = {self.fr_periodo_px:g} px · r_ideal = {r_ideal:.3f}')
            return
        
        imagenes = [ideal_fringe_image(info['u_p'], info['n_dot_l'], omega, omega_map, period_px=self.fr_periodo_px, u0_px=self.fr_u0_px, k=k, n_steps=self.fr_n_pasos,
                    ambient=self.fr_ambiente)[0] for k in range(1, self.fr_n_pasos + 1)]
        fase = psa_phase(imagenes, omega_map)
        self.visor_2d.show_phase(fase, omega_map, envuelta=True, title=f'Fase envuelta φ_ideal (PSA de {self.fr_n_pasos} pasos) · r_ideal = {r_ideal:.3f}')
        
    def _cargar_nivel_especular(self, indice):
        nombre = self.combo_fr_nivel.itemText(indice)
        nivel = NIVELES_ESPECULARES[nombre]
        self.campo_fr_g.setText(f"{nivel['g']:.2f}")
        self.campo_fr_o.setText(f"{nivel['o']:.2f}")
        self.campo_fr_sigma_esp.setText(f"{nivel['sigma']:g}")