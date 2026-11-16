import json
from dataclasses import dataclass, field, fields, asdict

VERSION_CONFIG = 1

@dataclass
class ConfigCamara:
    focal_mm: float = 12.0
    pixel_um: float = 4.65
    ancho: int = 1024
    alto: int = 768
    distancia_mm: float = 1000.0

@ dataclass
class ConfigProyector:
    focal_mm: float = 12.0
    pixel_um: float = 4.65
    ancho: int = 1024
    alto: int = 768
    distancia_mm: float = 1000.0
    angulo_deg: float = 12.0

@dataclass
class ConfigObjeto:
    alpha_deg: float = 0.0
    beta_deg: float = 0.0

@dataclass
class ConfigFranjas:
    periodo_px: float = 64.0
    n_pasos: int = 12
    u0_px: float = 512.5
    ambiente: float = 0.05

@dataclass
class ConfigDegradacion:
    sigma: float = 0.01
    semilla: int = 2026
    gamma: float = 2.6
    kappa: float = 60.0
    g: float = 1.10
    o: float = 0.30
    sigma_esp_px: float = 3.0

@dataclass
class ConfigEscena:
    camara: ConfigCamara = field(default_factory=ConfigCamara)
    proyector: ConfigProyector = field(default_factory=ConfigProyector)
    objeto: ConfigObjeto = field(default_factory=ConfigObjeto)
    franjas: ConfigFranjas = field(default_factory=ConfigFranjas)
    degradacion: ConfigDegradacion = field(default_factory=ConfigDegradacion)

    def validar(self):
        e = []
        c, p, f, d = self.camara, self.proyector, self.franjas, self.degradacion
        for nombre, v in (('cámara', c), ('proyector', p)):
            if v.focal_mm <= 0 or v.pixel_um <= 0:
                e.append(f'{nombre}: focal y tamaño del píxel deben ser > 0')
            if v.ancho < 1 or v.alto < 1:
                e.append(f'{nombre}: la resolución debe ser >= 1 px')
            if v.distancia_mm <= 0:
                e.append(f'{nombre}: la distancia debe ser > 0')
        if not 0 < p.angulo_deg < 90:
            e.append('proyector: el angúlo debe estar en (0, 90) grados')
        if f.periodo_px <= 0:
            e.append('franjas: el periodo Tp deber ser > 0')
        if f.n_pasos < 3:
            e.append('franjas: se necesitan al menos 3 pasos')
        if not 0 <= f.ambiente <= 1:
            e.append('franjas: I_amb debe estar en [0, 1]')
        if d.sigma < 0 or d.sigma_esp_px < 0:
            e.append('degradación: σ y σ_ℓ deben ser >= 0')
        if d.gamma <= 0 or d.kappa <= 0:
            e.append('degradación: γ y κs deben ser > 0')
        if d.g < 0 or d.o < 0:
            e.append('degradación: g y o deben ser >= 0')
        return e

    def a_diccionario(self):
        return {'version': VERSION_CONFIG, **asdict(self)}

    @classmethod
    def desde_diccionario(cls, datos):
        datos = dict(datos)
        version = datos.pop('version', VERSION_CONFIG)
        if version > VERSION_CONFIG:
            raise ValueError(f'Configuración de versión {version}; este progama soporta hasta {VERSION_CONFIG}')
        conocidas = {f.name for f in fields(cls)}
        extra = set(datos) - conocidas
        if extra:
            raise ValueError(f'Secciones desconocidas: {sorted(extra)}')
        partes = {}
        for f in fields(cls):
            base = f.default_factory()
            sec = datos.get(f.name, {})
            validos = {g.name for g in fields(base)}
            sobran = set(sec) - validos
            if sobran:
                raise ValueError(f'Campos desconocidos en "{f.name}": {sorted(sobran)}')
            for clave, valor in sec.items():
                tipo = type(getattr(base, clave))
                setattr(base, clave, tipo(valor))
            partes[f.name] = base
        return cls(**partes)
    
    def guardar_json(self, ruta):
        with open(ruta, 'w', encoding='utf-8') as fh:
            json.dump(self.a_diccionario(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def cargar_json(cls, ruta):
        with open(ruta, 'w', encoding='utf-8') as fh:
            return cls.desde_diccionario(json.load(fh))