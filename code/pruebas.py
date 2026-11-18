import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter

# ============================================================
# 1. GENERACIÓN DEL DATAFRAME SINTÉTICO
# ============================================================
np.random.seed(42)

n_comments = 300
fuentes = ['dmo', 'instagram', 'tripadvisor', 'google_reviews', 'blog']
pesos_fuentes = [0.15, 0.25, 0.25, 0.20, 0.15]
topicos = ['Playas', 'Cultura', 'Gastronomía', 'Precio', 'Seguridad', 'Servicio']
sentimientos = ['Positivo', 'Negativo']

fuentes_gen = np.random.choice(fuentes, size=n_comments, p=pesos_fuentes)

topicos_gen = []
for f in fuentes_gen:
    if f == 'dmo':
        t = np.random.choice(topicos, p=[0.30, 0.35, 0.15, 0.05, 0.05, 0.10])
    else:
        t = np.random.choice(topicos, p=[0.20, 0.10, 0.20, 0.25, 0.15, 0.10])
    topicos_gen.append(t)

sentimientos_gen = []
for i, f in enumerate(fuentes_gen):
    if f == 'dmo':
        s = np.random.choice(sentimientos, p=[0.85, 0.15])
    else:
        s = np.random.choice(sentimientos, p=[0.60, 0.40])
    sentimientos_gen.append(s)

umap_x, umap_y = [], []
for t in topicos_gen:
    idx = topicos.index(t)
    cx, cy = (idx * 2 - 5) * 1.5, np.sin(idx) * 3
    umap_x.append(cx + np.random.normal(0, 1.2))
    umap_y.append(cy + np.random.normal(0, 1.2))

sim_precio = np.random.beta(2, 5, size=n_comments)
for i, t in enumerate(topicos_gen):
    if t == 'Precio':
        sim_precio[i] = np.random.beta(7, 2)

longitud = np.random.randint(20, 300, size=n_comments)

# Textos ampliados para mejor extracción de n-gramas
textos_ejemplo = [
    "Las playas son increíbles y el servicio excelente en el hotel",
    "Muy caro para lo que ofrecen, no lo recomiendo para nada",
    "La cultura local es fascinante, muchos museos y buena comida",
    "Pésima atención en el hotel, muy sucio y mal organizado",
    "Gastronomía deliciosa, probé el ceviche y estaba fresco",
    "Me sentí inseguro caminando de noche por la zona turística",
    "Excelente relación calidad-precio, volvería sin dudarlo",
    "Los precios están fuera de control en temporada alta de verano",
    "El costo del estacionamiento fue una sorpresa desagradable",
    "Gran valor por tu dinero, el tour valió cada centavo invertido"
]

df = pd.DataFrame({
    'fuente': fuentes_gen,
    'topico': topicos_gen,
    'sentimiento': sentimientos_gen,
    'umap_x': umap_x,
    'umap_y': umap_y,
    'sim_precio': sim_precio,
    'longitud': longitud,
    'texto': [np.random.choice(textos_ejemplo) for _ in range(n_comments)]
})

df['tipo'] = df['fuente'].apply(lambda x: 'DMO' if x == 'dmo' else 'Percibido')

# ============================================================
# 2. PALETA "EJECUTIVA CONFIABLE" (Accesible y Profesional)
# ============================================================
C_TEXTO      = '#1A202C'  # Gris muy oscuro (casi negro)
C_TITULO     = '#2D3748'  # Gris pizarra oscuro
C_SUBTITULO  = '#4A5568'  # Gris medio
C_PARRAFO    = '#718096'  # Gris claro legible
C_PRIMARIO   = '#2B6CB0'  # Azul corporativo sólido (Unigramas)
C_SECUNDARIO = '#319795'  # Verde azulado / Teal (Bigramas)
C_TERCIARIO  = '#E2E8F0'  # Gris muy claro (fondos/bordes)
C_NARANJA    = '#DD6B20'  # Naranja quemado (Trigramas)
C_VERDE      = '#38A169'  # Verde éxito
C_ROJO       = '#E53E3E'  # Rojo alerta

PALETA_SECUENCIAL = [C_TERCIARIO, C_SECUNDARIO, C_PRIMARIO, C_TEXTO]

MAPA_TOPICOS = {
    'Playas': C_PRIMARIO, 'Cultura': C_SECUNDARIO, 'Gastronomía': C_NARANJA,
    'Precio': C_VERDE, 'Seguridad': C_ROJO, 'Servicio': C_SUBTITULO
}

def aplicar_estilo_ejecutivo(fig, title_text):
    fig.update_layout(
        title=dict(text=title_text, font=dict(color=C_TITULO, size=16, family="Arial, sans-serif")),
        font=dict(color=C_TEXTO, family="Arial, sans-serif"),
        template='plotly_white',
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        xaxis=dict(title_font=dict(color=C_SUBTITULO), tickfont=dict(color=C_PARRAFO)),
        yaxis=dict(title_font=dict(color=C_SUBTITULO), tickfont=dict(color=C_PARRAFO)),
        legend=dict(font=dict(color=C_TEXTO))
    )
    return fig

# ============================================================
# 3. FUNCIÓN AUXILIAR: EXTRACCIÓN DE N-GRAMAS
# ============================================================
def extraer_ngramas(textos, n):
    """Extrae n-gramas de una lista de textos usando Python puro."""
    ngramas = []
    for texto in textos:
        # Limpiar texto: minúsculas y solo palabras
        words = re.findall(r'\b\w+\b', str(texto).lower())
        # Filtrar palabras vacías comunes en español (stopwords básicas)
        stopwords = {'el', 'la', 'los', 'las', 'un', 'una', 'y', 'o', 'de', 'del', 'al', 'en', 'por', 'para', 'con', 'sin', 'muy', 'es', 'son', 'se', 'me', 'lo', 'que'}
        words = [w for w in words if w not in stopwords and len(w) > 2]
        
        for i in range(len(words) - n + 1):
            ngramas.append(" ".join(words[i:i+n]))
    return Counter(ngramas)

# ============================================================
# 4. PLOT 1: DIVERGING BAR CHART (Brecha Temática)
# ============================================================
def plot_diverging_bar(df):
    prop = df.groupby(['tipo', 'topico']).size().unstack(fill_value=0)
    prop_pct = prop.div(prop.sum(axis=0), axis=1) * 100
    gap = prop_pct.loc['Percibido'] - prop_pct.loc['DMO']
    gap = gap.sort_values(ascending=False)
    colors = [C_ROJO if v > 0 else C_PRIMARIO for v in gap.values]
    
    fig = go.Figure(go.Bar(
        x=gap.values, y=gap.index, orientation='h', marker_color=colors,
        hovertemplate="<b>%{y}</b><br>DMO: %{customdata[0]:.1f}%<br>Percibido: %{customdata[1]:.1f}%<br>Gap: %{x:+.1f}%<extra></extra>",
        customdata=[[prop_pct.loc['DMO', t], prop_pct.loc['Percibido', t]] for t in gap.index]
    ))
    fig.update_layout(xaxis_title="Diferencia (% Percibido - % DMO)", yaxis_title="Tópico", height=350, margin=dict(l=100, r=20, t=50, b=40))
    return aplicar_estilo_ejecutivo(fig, "Brecha Temática: DMO vs Imagen Percibida")

# ============================================================
# 5. PLOT 2: HEATMAP POR FUENTE
# ============================================================
def plot_heatmap_fuente(df):
    matriz = df.groupby(['fuente', 'topico']).size().unstack(fill_value=0)
    matriz_pct = matriz.div(matriz.sum(axis=1), axis=0) * 100
    fig = px.imshow(matriz_pct, labels=dict(x="Tópico", y="Fuente", color="Intensidad (%)"),
                    color_continuous_scale=PALETA_SECUENCIAL, aspect="auto", text_auto='.1f')
    fig.update_layout(height=350, margin=dict(l=100, r=20, t=50, b=40), coloraxis_colorbar=dict(tickfont=dict(color=C_TEXTO)))
    return aplicar_estilo_ejecutivo(fig, "Distribución de Tópicos por Fuente")

# ============================================================
# 6. PLOT 3: BARRAS AGRUPADAS
# ============================================================
def plot_barras_agrupadas(df):
    prop = df.groupby(['tipo', 'topico']).size().reset_index(name='count')
    total_por_tipo = prop.groupby('tipo')['count'].transform('sum')
    prop['pct'] = (prop['count'] / total_por_tipo) * 100
    fig = px.bar(prop, x='topico', y='pct', color='tipo', barmode='group',
                 color_discrete_map={'DMO': C_PRIMARIO, 'Percibido': C_SECUNDARIO}, hover_data=['count'])
    fig.update_layout(height=350, margin=dict(l=60, r=20, t=50, b=40))
    return aplicar_estilo_ejecutivo(fig, "Tópicos Principales: DMO vs Percibido")

# ============================================================
# 7. PLOT 4: SCATTER UMAP
# ============================================================
def plot_scatter_umap(df):
    df['simbolo'] = df['tipo'].apply(lambda x: 'star' if x == 'DMO' else 'circle')
    fig = px.scatter(df, x='umap_x', y='umap_y', color='topico', symbol='tipo', size='longitud', size_max=15,
                     hover_data=['fuente', 'sentimiento', 'texto'], color_discrete_map=MAPA_TOPICOS,
                     symbol_map={'DMO': 'star', 'Percibido': 'circle'})
    fig.update_traces(marker=dict(line=dict(width=1, color=C_TEXTO)))
    fig.update_layout(height=500, margin=dict(l=40, r=20, t=50, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return aplicar_estilo_ejecutivo(fig, "Mapa Semántico de Comentarios (UMAP)")

# ============================================================
# 8. PLOT 5: SCATTER PRECIO/VALOR
# ============================================================
def plot_scatter_precio(df):
    df_sample = df.sample(n=150, random_state=42)
    fig = px.scatter(df_sample, x='sim_precio', y=np.random.randn(len(df_sample)), color='tipo', size='sim_precio', size_max=20,
                     color_discrete_map={'DMO': C_PRIMARIO, 'Percibido': C_NARANJA}, hover_data=['fuente', 'sentimiento', 'texto'])
    fig.add_vline(x=0.6, line_dash="dash", line_color=C_ROJO, annotation_text="Umbral relevancia", annotation_font=dict(color=C_ROJO, size=12))
    fig.update_layout(height=300, showlegend=True)
    return aplicar_estilo_ejecutivo(fig, "Proximidad Semántica: Concepto 'Precio/Valor'")

# ============================================================
# 9. PLOT 6: NUBE DE PALABRAS ACCESIBLE (Simulada con Plotly)
# ============================================================
def plot_wordcloud_ejecutivo(df):
    """Simula una nube de palabras usando Scatter, escalando por frecuencia y coloreando por tipo de n-grama."""
    # Obtener top 40 n-gramas combinados (1, 2 y 3 palabras)
    uni = extraer_ngramas(df['texto'], 1).most_common(40)
    bi = extraer_ngramas(df['texto'], 2).most_common(30)
    tri = extraer_ngramas(df['texto'], 3).most_common(20)
    
    todos = []
    for palabra, freq in uni: todos.append((palabra, freq, 1))
    for palabra, freq in bi: todos.append((palabra, freq, 2))
    for palabra, freq in tri: todos.append((palabra, freq, 3))
    
    # Ordenar por frecuencia y tomar top 50
    todos = sorted(todos, key=lambda x: x[1], reverse=True)[:50]
    max_freq = todos[0][1]
    
    x_vals, y_vals, textos, sizes, colors = [], [], [], [], []
    
    # Distribución pseudo-aleatoria controlada para evitar solapamiento excesivo
    np.random.seed(99)
    for i, (palabra, freq, n_len) in enumerate(todos):
        x_vals.append(np.random.uniform(5, 95))
        y_vals.append(np.random.uniform(5, 95))
        textos.append(palabra.title())
        # Tamaño de fuente entre 12 y 36 según frecuencia
        sizes.append(12 + int((freq / max_freq) * 24))
        # Color según longitud del n-grama (Paleta Ejecutiva)
        if n_len == 1: colors.append(C_PRIMARIO)
        elif n_len == 2: colors.append(C_SECUNDARIO)
        else: colors.append(C_NARANJA)

    fig = go.Figure(go.Scatter(
        x=x_vals, y=y_vals, mode='text', text=textos,
        textfont=dict(size=sizes, color=colors, family="Arial, sans-serif"),
        hovertext=[f"Frecuencia: {f}" for _, f, _ in todos],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
        title=dict(text="Nube de Conceptos Clave (Color: Azul=1 palabra, Teal=2, Naranja=3)", 
                   font=dict(color=C_TITULO, size=14), x=0.5)
    )
    return fig

# ============================================================
# 10. PLOT 7: N-GRAMAS POR SEPARADO (Unigrama, Bigrama, Trigrama)
# ============================================================
def plot_ngramas_separados(df):
    """Muestra los top 5 de unigramas, bigramas y trigramas en gráficos horizontales separados."""
    uni = extraer_ngramas(df['texto'], 1).most_common(5)
    bi = extraer_ngramas(df['texto'], 2).most_common(5)
    tri = extraer_ngramas(df['texto'], 3).most_common(5)
    
    fig = make_subplots(rows=3, cols=1, subplot_titles=("Top 5 Unigramas", "Top 5 Bigramas", "Top 5 Trigramas"),
                        vertical_spacing=0.15, horizontal_spacing=0.1)
    
    # Unigramas (Azul)
    fig.add_trace(go.Bar(x=[v for k,v in uni], y=[k.title() for k,v in uni], orientation='h', 
                         marker_color=C_PRIMARIO, name='Unigramas', text=[v for k,v in uni], textposition='auto'), row=1, col=1)
    
    # Bigramas (Teal)
    fig.add_trace(go.Bar(x=[v for k,v in bi], y=[k.title() for k,v in bi], orientation='h', 
                         marker_color=C_SECUNDARIO, name='Bigramas', text=[v for k,v in bi], textposition='auto'), row=2, col=1)
    
    # Trigramas (Naranja)
    fig.add_trace(go.Bar(x=[v for k,v in tri], y=[k.title() for k,v in tri], orientation='h', 
                         marker_color=C_NARANJA, name='Trigramas', text=[v for k,v in tri], textposition='auto'), row=3, col=1)
    
    fig.update_layout(
        height=500,
        margin=dict(l=120, r=40, t=60, b=40),
        showlegend=False,
        title=dict(text="Análisis Detallado de N-gramas por Longitud", font=dict(color=C_TITULO, size=16), x=0.5)
    )
    
    # Estilizar ejes
    for i in range(1, 4):
        fig.update_yaxes(title_text="", tickfont=dict(color=C_TEXTO, size=11), row=i, col=1)
        fig.update_xaxes(title_text="Frecuencia", tickfont=dict(color=C_PARRAFO), row=i, col=1)
        
    return fig

# ============================================================
# 11. GENERAR TODOS LOS PLOTS Y EXPORTAR A HTML
# ============================================================
if __name__ == "__main__":
    
    plots = {
        "01_diverging_bar": plot_diverging_bar(df),
        "02_heatmap_fuente": plot_heatmap_fuente(df),
        "03_barras_agrupadas": plot_barras_agrupadas(df),
        "04_scatter_umap": plot_scatter_umap(df),
        "05_scatter_precio": plot_scatter_precio(df),
        "06_wordcloud": plot_wordcloud_ejecutivo(df),
        "07_ngramas_separados": plot_ngramas_separados(df)
    }
    
    for nombre, fig in plots.items():
        ruta = f"{nombre}.html"
        fig.write_html(ruta, include_plotlyjs='cdn', full_html=True)
        print(f"✅ Guardado: {ruta}")
    
    # Dashboard completo en un solo HTML (Ahora 4 filas x 2 columnas para acomodar todo)
    fig_dashboard = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Brecha Temática (Diverging)", "Distribución de Tópicos por Fuente",
            "Tópicos DMO vs Percibido", "Mapa Semántico UMAP",
            "Similitud 'Precio/Valor'", "Nube de Conceptos Clave",
            "Análisis Detallado de N-gramas (1, 2 y 3 palabras)", ""
        ),
        specs=[
            [{"type": "bar"}, {"type": "heatmap"}],
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}], # Wordcloud usa tipo scatter
            [{"type": "bar", "colspan": 2}, None]       # N-gramas separados ocupa todo el ancho
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        row_heights=[0.2, 0.2, 0.25, 0.35]
    )
    
    # Mapeo manual de plots a posiciones del dashboard
    posiciones = [
        ("01_diverging_bar", 1, 1),
        ("02_heatmap_fuente", 1, 2),
        ("03_barras_agrupadas", 2, 1),
        ("04_scatter_umap", 2, 2),
        ("05_scatter_precio", 3, 1),
        ("06_wordcloud", 3, 2),
        ("07_ngramas_separados", 4, 1) # Colspan 2 manejado arriba
    ]
    
    for nombre, row, col in posiciones:
        fig = plots[nombre]
        for trace in fig.data:
            fig_dashboard.add_trace(trace, row=row, col=col)
    
    # Actualizar títulos de subplot con el color ejecutivo
    for i, annotation in enumerate(fig_dashboard.layout.annotations):
        annotation.font.color = C_TITULO
        annotation.font.size = 13

    fig_dashboard.update_layout(
        title=dict(text="Dashboard Ejecutivo: Análisis DMO vs Imagen Percibida", font=dict(color=C_TITULO, size=22)),
        height=1600,
        template='plotly_white',
        font=dict(color=C_TEXTO),
        showlegend=True
    )
    
    fig_dashboard.write_html("dashboard_completo.html", include_plotlyjs='cdn', full_html=True)
    print("✅ Dashboard completo guardado: dashboard_completo.html")