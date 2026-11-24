import re
from pathlib import Path

from PySide6.QtCore import QLocale
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QVBoxLayout)

class DialogoBarrido(QDialog):
    def __init__(self, padre=None, nombre_modelo='modelo'):
        super().__init__(padre)
        self.setWindowTitle('Barrido de orientaciones')
        self.setMinimumWidth(460)

        self.campo_modelo = QLineEdit(nombre_modelo)
        self.campo_carpeta = QLineEdit()
        self.campo_carpeta.setPlaceholderText('Carpeta donde se guardarán las escenas')
        boton_carpeta = QPushButton('Examinar...')
        boton_carpeta.clicked.connect(self._elegir_carpeta)
        fila_carpeta = QHBoxLayout()
        fila_carpeta.addWidget(self.campo_carpeta)
        fila_carpeta.addWidget(boton_carpeta)

        self.n_alpha = self._spin_entero(8)
        self.paso_alpha = self._spin_decimal(45.0, 0.1, 360.0)
        self.alpha0 = self._spin_decimal(0.0, -360.0, 360.0)
        self.n_beta = self._spin_entero(24)
        self.paso_beta = self._spin_decimal(15.0, 0.1, 360.0)
        self.beta0 = self._spin_decimal(0.0, -360.0, 360.0)

        self.check_planificacion = QCheckBox('Seleccionar vistas con planificación Fibonacci')
        self.num_candidatas = QSpinBox()
        self.num_candidatas.setRange(4, 256)
        self.num_candidatas.setValue(48)
        self.max_vistas = QSpinBox()
        self.max_vistas.setRange(1, 64)
        self.max_vistas.setValue(8)
        self.porcentaje_malla = QSpinBox()
        self.porcentaje_malla.setRange(1, 100)
        self.porcentaje_malla.setValue(10)
        self.incidencia_max = self._spin_decimal(60.0, 0.0, 90.0)
        self.solapamiento_minimo = QSpinBox()
        self.solapamiento_minimo.setRange(0, 100)
        self.solapamiento_minimo.setValue(15)

        opciones_plan = QFormLayout()
        opciones_plan.addRow('Vistas candidatas:', self.num_candidatas)
        opciones_plan.addRow('Máximo de vistas seleccionadas:', self.max_vistas)
        opciones_plan.addRow('Malla conservada para el análisis (%):', self.porcentaje_malla)
        opciones_plan.addRow('Ángulo de incidencia máximo (°):', self.incidencia_max)
        opciones_plan.addRow('Solapamiento mínimo entre vistas (%):', self.solapamiento_minimo)
        self.grupo_planificacion = QGroupBox()
        self.grupo_planificacion.setLayout(opciones_plan)
        self.grupo_planificacion.setEnabled(False)
        self.check_planificacion.toggled.connect(self.grupo_planificacion.setEnabled)
        nota_plan = QLabel(
            'Evalúa una malla reducida con pruebas geométricas de menor coste; '
            'las vistas elegidas se simulan después con la malla original. '
            'Al activarla, la esfera de Fibonacci reemplaza la rejilla α/β anterior.'
        )
        nota_plan.setWordWrap(True)

        self.check_niveles = QCheckBox('Generar los cinco niveles de degradación')
        nota_niveles = QLabel('low, medium, moderate, strong y very_strong. Si la desmarcas, se genera un solo nivel, el de la configuración actual.')
        nota_niveles.setWordWrap(True)

        self.check_reanudar = QCheckBox('Omitir escenas ya exportadas (reanudar)')
        self.check_reanudar.setChecked(True)
        nota = QLabel('Si la desmarcas, las escenas que ya existan se sobrescriben.')
        nota.setWordWrap(True)
        self.label_total = QLabel()

        formulario = QFormLayout()
        formulario.addRow('Nombre del modelo:', self.campo_modelo)
        formulario.addRow('Carpeta de salida:', fila_carpeta)
        formulario.addRow('Rotación X (α): cantidad', self.n_alpha)
        formulario.addRow('Rotación X (α): paso (°)', self.paso_alpha)
        formulario.addRow('Rotación X (α): inicio (°)', self.alpha0)
        formulario.addRow('Rotación Y (β): cantidad', self.n_beta)
        formulario.addRow('Rotación Y (β): paso (°)', self.paso_beta)
        formulario.addRow('Rotación Y (β): inicio (°)', self.beta0)

        botones = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        botones.button(QDialogButtonBox.StandardButton.Ok).setText('Iniciar barrido')
        botones.button(QDialogButtonBox.StandardButton.Cancel).setText('Cancelar')
        botones.accepted.connect(self._validar_y_aceptar)
        botones.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(formulario)
        layout.addWidget(self.check_planificacion)
        layout.addWidget(self.grupo_planificacion)
        layout.addWidget(nota_plan)
        layout.addWidget(self.check_niveles)
        layout.addWidget(nota_niveles)
        layout.addWidget(self.check_reanudar)
        layout.addWidget(nota)
        layout.addWidget(self.label_total)
        layout.addWidget(botones)

        self.n_alpha.valueChanged.connect(self._actualizar_total)
        self.n_beta.valueChanged.connect(self._actualizar_total)
        self._actualizar_total()
    
    @staticmethod
    def _spin_entero(valor):
        spin = QSpinBox()
        spin.setRange(1, 360)
        spin.setValue(valor)
        return spin

    @staticmethod
    def _spin_decimal(valor, minimo, maximo):
        spin = QDoubleSpinBox()
        spin.setLocale(QLocale.c())
        spin.setRange(minimo, maximo)
        spin.setDecimals(2)
        spin.setValue(valor)
        return spin

    def _actualizar_total(self):
        total = self.n_alpha.value() * self.n_beta.value()
        self.label_total.setText(f'Total: {total:,} escenas')

    def _elegir_carpeta(self):
        carpeta = QFileDialog.getExistingDirectory(self, 'Carpeta de salida')
        if carpeta:
            self.campo_carpeta.setText(carpeta)

    def _validar_y_aceptar(self):
        modelo = self.campo_modelo.text().strip()
        if not modelo or re.search(r'[\\/:*?"<>|]', modelo):
            QMessageBox.warning(self, 'Barrido', 'Escribe un nombre de modelo válido (sin \\ / : * ? " < > |).')
            return
        if not Path(self.campo_carpeta.text()).is_dir():
            QMessageBox.warning(self, 'Barrido', 'Elige una carpeta de salida que exista.')
            return
        self.accept()

    def valores(self):
        return {
            'modelo': self.campo_modelo.text().strip(), 
            'carpeta': self.campo_carpeta.text(),
            'n_alpha': self.n_alpha.value(), 'paso_alpha': self.paso_alpha.value(), 'alpha0': self.alpha0.value(),
            'n_beta': self.n_beta.value(), 'paso_beta': self.paso_beta.value(), 'beta0': self.beta0.value(),
            'planificar': self.check_planificacion.isChecked(),
            'num_candidatas': self.num_candidatas.value(),
            'max_vistas': self.max_vistas.value(),
            'porcentaje_malla': self.porcentaje_malla.value(),
            'incidencia_max': self.incidencia_max.value(),
            'solapamiento_minimo': self.solapamiento_minimo.value() / 100.0,
            'cinco_niveles': self.check_niveles.isChecked(),
            'reanudar': self.check_reanudar.isChecked(),
        }