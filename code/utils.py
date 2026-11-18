"""
Módulo de utilidades para estilos y paletas de colores accesibles.
Contiene paletas adaptadas para visión normal, protanopia y deuteranopia.
"""

from typing import Dict, List, Any
import plotly.graph_objects as go


# ============================================================
# PALETAS DE COLORES COMPLETAS (con adaptaciones para daltonismo)
# ============================================================
COLOR_PALETTES = {
    "ejecutiva": {
         "normal": {
            "texto": "#1A202C",
            "titulo": "#2D3748",
            "subtitulo": "#4A5568",
            "parrafo": "#637083",  # ← CAMBIADO de #718096
            "primario": "#2B6CB0",
            "secundario": "#319795",
            "terciario": "#E2E8F0",
            "complementarios": ["#DD6B20", "#38A169", "#E53E3E"],
            "fondo": "#FFFFFF",
            "borde": "#CBD5E0"
        },
        "protanopia": {
            "texto": "#1A202C",
            "titulo": "#2D3748",
            "subtitulo": "#4A5568",
            "parrafo": "#637083",  # ← CAMBIADO de #718096
            "primario": "#2B6CB0",
            "secundario": "#319795",
            "terciario": "#E2E8F0",
            "complementarios": ["#D4A574", "#38A169", "#C53030"],
            "fondo": "#FFFFFF",
            "borde": "#CBD5E0"
        },
        "deuteranopia": {
            "texto": "#1A202C",
            "titulo": "#2D3748",
            "subtitulo": "#4A5568",
            "parrafo": "#637083",  # ← CAMBIADO de #718096
            "primario": "#2B6CB0",
            "secundario": "#319795",
            "terciario": "#E2E8F0",
            "complementarios": ["#DD6B20", "#2F855A", "#C53030"],
            "fondo": "#FFFFFF",
            "borde": "#CBD5E0"
        }
    },
    
      "naturaleza": {
        "normal": {
            "texto": "#264653",
            "titulo": "#2A9D8F",
            "subtitulo": "#A68A1F",  # ← CAMBIADO de #E9C46A (dorado oscuro)
            "parrafo": "#555B6E",
            "primario": "#2A9D8F",
            "secundario": "#E9C46A",  # El amarillo claro se mantiene como acento gráfico
            "terciario": "#F4A261",
            "complementarios": ["#E76F51", "#264653", "#2B2D42"],
            "fondo": "#FFFFFF",
            "borde": "#D3D3D3"
        },
        "protanopia": {
            "texto": "#264653",
            "titulo": "#2A9D8F",
            "subtitulo": "#9A7B1F",  # ← CAMBIADO de #D4A853 (dorado oscuro adaptado)
            "parrafo": "#555B6E",
            "primario": "#2A9D8F",
            "secundario": "#D4A853",  # El amarillo se mantiene como acento gráfico
            "terciario": "#E09959",
            "complementarios": ["#D65A31", "#264653", "#2B2D42"],
            "fondo": "#FFFFFF",
            "borde": "#D3D3D3"
        },
        "deuteranopia": {
            "texto": "#264653",
            "titulo": "#2A9D8F",
            "subtitulo": "#A68A1F",  # ← CAMBIADO de #E9C46A (dorado oscuro)
            "parrafo": "#555B6E",
            "primario": "#2A9D8F",
            "secundario": "#E9C46A",  # El amarillo se mantiene como acento gráfico
            "terciario": "#F4A261",
            "complementarios": ["#E76F51", "#264653", "#2B2D42"],
            "fondo": "#FFFFFF",
            "borde": "#D3D3D3"
        }
    },
    
    "accesible": {
        "normal": {
            "texto": "#00224E",
            "titulo": "#00224E",
            "subtitulo": "#3B5B85",
            "parrafo": "#4A5568",
            "primario": "#184C7A",
            "secundario": "#7A94B0",
            "terciario": "#C4C4B0",
            "complementarios": ["#FFF3B3", "#00224E", "#1B3A5C"],
            "fondo": "#FFFFFF",
            "borde": "#B0B0B0"
        },
        "protanopia": {
            "texto": "#00224E",
            "titulo": "#00224E",
            "subtitulo": "#3B5B85",
            "parrafo": "#4A5568",
            "primario": "#184C7A",
            "secundario": "#7A94B0",
            "terciario": "#C4C4B0",
            "complementarios": ["#FFF3B3", "#00224E", "#1B3A5C"],
            "fondo": "#FFFFFF",
            "borde": "#B0B0B0"
        },
        "deuteranopia": {
            "texto": "#00224E",
            "titulo": "#00224E",
            "subtitulo": "#3B5B85",
            "parrafo": "#4A5568",
            "primario": "#184C7A",
            "secundario": "#7A94B0",
            "terciario": "#C4C4B0",
            "complementarios": ["#FFF3B3", "#00224E", "#1B3A5C"],
            "fondo": "#FFFFFF",
            "borde": "#B0B0B0"
        }
    },
    
    "tecnologica": {
        "normal": {
            "texto": "#1A202C",
            "titulo": "#553C9A",
            "subtitulo": "#805AD5",
            "parrafo": "#4A5568",
            "primario": "#6B46C1",
            "secundario": "#D53F8C",
            "terciario": "#ED8936",
            "complementarios": ["#38B2AC", "#E53E3E", "#ECC94B"],
            "fondo": "#FFFFFF",
            "borde": "#D6BCFA"
        },
        "protanopia": {
            "texto": "#1A202C",
            "titulo": "#553C9A",
            "subtitulo": "#805AD5",
            "parrafo": "#4A5568",
            "primario": "#6B46C1",
            "secundario": "#B83280",
            "terciario": "#ED8936",
            "complementarios": ["#38B2AC", "#C53030", "#ECC94B"],
            "fondo": "#FFFFFF",
            "borde": "#D6BCFA"
        },
        "deuteranopia": {
            "texto": "#1A202C",
            "titulo": "#553C9A",
            "subtitulo": "#805AD5",
            "parrafo": "#4A5568",
            "primario": "#6B46C1",
            "secundario": "#D53F8C",
            "terciario": "#ED8936",
            "complementarios": ["#38B2AC", "#E53E3E", "#ECC94B"],
            "fondo": "#FFFFFF",
            "borde": "#D6BCFA"
        }
    },
    
    "hospitalidad": {
        "normal": {
            "texto": "#2D3748",
            "titulo": "#C05621",
            "subtitulo": "#DD6B20",
            "parrafo": "#4A5568",
            "primario": "#DD6B20",
            "secundario": "#2B6CB0",
            "terciario": "#ECC94B",
            "complementarios": ["#38A169", "#2D3748", "#744210"],
            "fondo": "#FFFFFF",
            "borde": "#F6AD55"
        },
        "protanopia": {
            "texto": "#2D3748",
            "titulo": "#C05621",
            "subtitulo": "#DD6B20",
            "parrafo": "#4A5568",
            "primario": "#DD6B20",
            "secundario": "#2B6CB0",
            "terciario": "#ECC94B",
            "complementarios": ["#38A169", "#2D3748", "#744210"],
            "fondo": "#FFFFFF",
            "borde": "#F6AD55"
        },
        "deuteranopia": {
            "texto": "#2D3748",
            "titulo": "#C05621",
            "subtitulo": "#DD6B20",
            "parrafo": "#4A5568",
            "primario": "#DD6B20",
            "secundario": "#2B6CB0",
            "terciario": "#ECC94B",
            "complementarios": ["#2F855A", "#2D3748", "#744210"],
            "fondo": "#FFFFFF",
            "borde": "#F6AD55"
        }
    }
}


# ============================================================
# FUNCIONES AUXILIARES PARA PALETAS
# ============================================================
def get_palette(palette_name: str, vision_type: str = "normal") -> Dict[str, Any]:
    """
    Obtiene una paleta de colores específica.
    
    Args:
        palette_name: Nombre de la paleta (ejecutiva, naturaleza, accesible, tecnologica, hospitalidad)
        vision_type: Tipo de visión (normal, protanopia, deuteranopia)
    
    Returns:
        Diccionario con todos los colores de la paleta
    """
    if palette_name not in COLOR_PALETTES:
        raise ValueError(f"Paleta '{palette_name}' no encontrada. Opciones: {list(COLOR_PALETTES.keys())}")
    
    if vision_type not in ["normal", "protanopia", "deuteranopia"]:
        raise ValueError(f"Tipo de visión '{vision_type}' no válido. Opciones: normal, protanopia, deuteranopia")
    
    return COLOR_PALETTES[palette_name][vision_type]


def get_plotly_colorscale(palette_name: str, vision_type: str = "normal") -> List[str]:
    """
    Genera una escala de colores para Plotly basada en la paleta seleccionada.
    
    Args:
        palette_name: Nombre de la paleta
        vision_type: Tipo de visión
    
    Returns:
        Lista de colores para usar en color_continuous_scale de Plotly
    """
    palette = get_palette(palette_name, vision_type)
    return [palette["terciario"], palette["secundario"], palette["primario"]]


def get_categorical_colors(palette_name: str, vision_type: str = "normal", n_colors: int = 5) -> List[str]:
    """
    Obtiene colores categóricos para gráficas discretas.
    
    Args:
        palette_name: Nombre de la paleta
        vision_type: Tipo de visión
        n_colors: Número de colores necesarios
    
    Returns:
        Lista de colores para series categóricas
    """
    palette = get_palette(palette_name, vision_type)
    colors = [palette["primario"], palette["secundario"], palette["terciario"]]
    colors.extend(palette["complementarios"])
    return colors[:n_colors]


def _hex_to_rgb(value: str):
    value = value.lstrip("#")
    if len(value) != 6:
        return (0, 0, 0)
    try:
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return (0, 0, 0)


def _closest_color(candidates: List[str], target: str) -> str:
    """Devuelve el color de `candidates` más cercano (en RGB) a `target`."""
    if not candidates:
        return target

    tr, tg, tb = _hex_to_rgb(target)
    mejor = candidates[0]
    mejor_dist = None

    for color in candidates:
        r, g, b = _hex_to_rgb(color)
        dist = (r - tr) ** 2 + (g - tg) ** 2 + (b - tb) ** 2
        if mejor_dist is None or dist < mejor_dist:
            mejor_dist = dist
            mejor = color

    return mejor


def palette_to_plot_colors(
    palette_name: str,
    vision_type: str = "normal",
) -> Dict[str, Any]:
    """
    Convierte una paleta de `utils` a los nombres de color que usan los
    gráficos del proyecto (primary/secondary/... + positive/negative).

    Permite que todos los HTML compartan la misma gama de colores.
    """
    palette = get_palette(palette_name, vision_type)
    complementarios = list(palette.get("complementarios", []))

    def _pick(index: int, fallback: str) -> str:
        if len(complementarios) > index:
            return complementarios[index]
        return fallback

    # Para sentimiento se eligen los acentos más cercanos al verde (positivo)
    # y al rojo/naranja (negativo) para no cambiar la semántica de las gráficas.
    candidatos = complementarios + [palette["primario"], palette["secundario"]]

    return {
        "primary": palette["primario"],
        "secondary": palette["secundario"],
        "tertiary": _pick(0, palette["secundario"]),
        "quaternary": _pick(1, palette["primario"]),
        "background": palette.get("fondo", "#FFFFFF"),
        "text": palette.get("texto", "#1A202C"),
        "title": palette.get("titulo", "#2D3748"),
        "positive": _closest_color(candidatos, "#009E73"),
        "negative": _closest_color(candidatos, "#D55E00"),
        "categorical": get_categorical_colors(palette_name, vision_type, n_colors=8),
    }


# ============================================================
# CONSTANTES DE ESTILO EJECUTIVO (para compatibilidad con código existente)
# ============================================================
# Estos valores se usan como fallback si no se especifica una paleta
C_TITULO = "#1f77b4"
C_SUBTITULO = "#1f77b4"
C_TEXTO = "#333333"
C_PARRAFO = "#555555"

# Paletas accesibles para daltonismo (nombres para CLI)
PALETAS_ACCESIBLES = ['ejecutiva', 'naturaleza', 'accesible', 'tecnologica', 'hospitalidad']


# ============================================================
# FUNCIONES DE ESTILO PARA PLOTLY
# ============================================================
def aplicar_estilo_ejecutivo(
    fig: go.Figure, 
    title_text: str,
    palette_name: str = "ejecutiva",
    vision_type: str = "normal"
) -> go.Figure:
    """
    Aplica formato ejecutivo, accesible y consistente a figuras de Plotly.
    
    Args:
        fig: Figura de Plotly a estilizar
        title_text: Texto del título
        palette_name: Nombre de la paleta de colores
        vision_type: Tipo de visión (normal, protanopia, deuteranopia)
    
    Returns:
        Figura estilizada
    """
    palette = get_palette(palette_name, vision_type)
    
    fig.update_layout(
        title=dict(
            text=title_text, 
            font=dict(color=palette["titulo"], size=16, family="Arial, sans-serif")
        ),
        font=dict(color=palette["texto"], family="Arial, sans-serif"),
        template='plotly_white',
        plot_bgcolor=palette["fondo"],
        paper_bgcolor=palette["fondo"],
        xaxis=dict(
            title_font=dict(color=palette["subtitulo"]), 
            tickfont=dict(color=palette["parrafo"])
        ),
        yaxis=dict(
            title_font=dict(color=palette["subtitulo"]), 
            tickfont=dict(color=palette["parrafo"])
        ),
        legend=dict(font=dict(color=palette["texto"]))
    )
    return fig


def aplicar_estilo_a_dataframe_table(
    fig: go.Figure,
    palette_name: str = "ejecutiva",
    vision_type: str = "normal"
) -> go.Figure:
    """
    Aplica estilo a tablas en Plotly.
    
    Args:
        fig: Figura de tabla de Plotly
        palette_name: Nombre de la paleta
        vision_type: Tipo de visión
    
    Returns:
        Figura estilizada
    """
    palette = get_palette(palette_name, vision_type)
    
    fig.update_layout(
        font=dict(color=palette["texto"], family="Arial, sans-serif"),
        template='plotly_white',
        plot_bgcolor=palette["fondo"],
        paper_bgcolor=palette["fondo"]
    )
    return fig


# ============================================================
# VALIDACIÓN DE ACCESIBILIDAD
# ============================================================
def validar_contraste(color_fg: str, color_bg: str = "#FFFFFF") -> float:
    """
    Calcula el ratio de contraste entre dos colores (WCAG 2.1).
    
    Args:
        color_fg: Color de primer plano (hex)
        color_bg: Color de fondo (hex)
    
    Returns:
        Ratio de contraste (debe ser >= 4.5 para texto normal, >= 3.0 para texto grande)
    """
    def hex_to_rgb(hex_color: str) -> tuple:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def relative_luminance(rgb: tuple) -> float:
        r, g, b = [x / 255.0 for x in rgb]
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    l1 = relative_luminance(hex_to_rgb(color_fg))
    l2 = relative_luminance(hex_to_rgb(color_bg))
    
    lighter = max(l1, l2)
    darker = min(l1, l2)
    
    return (lighter + 0.05) / (darker + 0.05)


def validar_paleta_accesible(palette_name: str) -> Dict[str, bool]:
    """
    Valida que una paleta cumpla con los estándares de accesibilidad WCAG 2.1.
    
    Args:
        palette_name: Nombre de la paleta a validar
    
    Returns:
        Diccionario con resultados de validación
    """
    palette = get_palette(palette_name, "normal")
    bg = palette["fondo"]
    
    resultados = {
        "texto_vs_fondo": validar_contraste(palette["texto"], bg) >= 4.5,
        "titulo_vs_fondo": validar_contraste(palette["titulo"], bg) >= 3.0,
        "subtitulo_vs_fondo": validar_contraste(palette["subtitulo"], bg) >= 3.0,
        "parrafo_vs_fondo": validar_contraste(palette["parrafo"], bg) >= 4.5,
    }
    
    return resultados


if __name__ == "__main__":
    # Prueba rápida de validación
    print("Validando accesibilidad de paletas...")
    for palette_name in PALETAS_ACCESIBLES:
        print(f"\n{palette_name.upper()}:")
        resultados = validar_paleta_accesible(palette_name)
        for check, passed in resultados.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")

