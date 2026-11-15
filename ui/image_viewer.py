import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg, NavigationToolbar2QT)
from PySide6.QtWidgets import QWidget, QVBoxLayout

class ImageViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure = Figure(facecolor='#FFFFFF')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def show_depth(self, depth, mask, title='', cbar_label='Profundidad Z (mm)'):
        alto, ancho = depth.shape

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not mask.any():
            ax.text(0.5, 0.5, 'Sin píxeles visibles\ncon la configuración actual', ha='center', va='center', fontsize=13, transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw_idle()
            return
        
        imagen = np.ma.masked_where(~mask, depth)

        cmap = matplotlib.colormaps['gray_r'].copy()
        cmap.set_bad('#FFFFFF')

        im = ax.imshow(imagen, cmap=cmap, interpolation='nearest', extent=(0.5, ancho + 0.5, alto + 0.5, 0.5))

        ax.set_xlabel('Columna u (px)')
        ax.set_ylabel('Fila v (px)')
        ax.set_title(title, fontsize=11)

        cbar = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
        
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def show_normals(self, normals, mask, title=''):
        alto, ancho = mask.shape

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not mask.any():
            ax.text(0.5, 0.5, 'Sin píxeles visibles\ncon la configuración actual', ha='center', va='center', fontsize=13, transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw_idle()
            return

        fondo = np.array([0xE2, 0xE8, 0xF0]) / 255.0
        rgb = np.tile(fondo, (alto, ancho, 1))
        rgb[mask] = np.clip((normals[mask] + 1.0) / 2.0, 0.0, 1.0)

        ax.imshow(rgb, interpolation='nearest', extent=(0.5, ancho + 0.5, alto + 0.5, 0.5))
        ax.set_xlabel('Columna u (px)')
        ax.set_ylabel('Fila v (px)')
        ax.set_title(title, fontsize=11)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def show_categories(self, labels, names, colors, title=''):
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.patches import Patch

        alto, ancho = labels.shape

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        n = len(names)
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(-0.5, n + 0.5, 1.0), cmap.N)
        ax.imshow(labels, cmap=cmap, norm=norm, interpolation='nearest', extent=(0.5, ancho + 0.5, alto + 0.5, 0.5))

        total = max(int((labels > 0).sum()), 1)
        handles = []
        for k in range(1, n):
            cuenta = int((labels == k).sum())
            if cuenta > 0:
                handles.append(Patch(facecolor=colors[k], edgecolor='#475569', label=f'{names[k]}: {cuenta:,} ({100.0 * cuenta / total:.1f} %)'))

        if handles:
            ax.legend(handles=handles, loc='upper right', fontsize=9, framealpha=0.95)

        ax.set_xlabel('Columna u (px)')
        ax.set_ylabel('Fila v (px)')
        ax.set_title(title, fontsize=11)
        
        self.figure.tight_layout()
        self.canvas.draw_idle()
    
    def show_intensity(self, image, title='', cbar_label='Intensidad normalizada', vmax=1.0):
        alto, ancho = image.shape

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        im = ax.imshow(image, cmap='gray', vmin=0.0, vmax=vmax, interpolation='nearest', extent=(0.5, ancho + 0.5, alto + 0.5, 0.5))

        ax.set_xlabel('Columna u (px)')
        ax.set_ylabel('Fila v (px)')
        ax.set_title(title, fontsize=11)

        cbar = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

        self.figure.tight_layout()
        self.canvas.draw_idle()