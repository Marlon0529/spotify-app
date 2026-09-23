import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# 1. Configuración de la Interfaz
# ---------------------------------------------------------
st.set_page_config(
    page_title="Spotify Vibe Explorer",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Spotify Vibe Explorer & Cluster Predictor")
st.markdown("""
Esta aplicación permite analizar y clasificar la "vibra" de canciones de Spotify usando **K-Means Clustering** 
y **PCA (Reducción de Dimensionalidad)** desarrollados en el laboratorio de Machine Learning.
""")

# ---------------------------------------------------------
# 2. Carga de Artefactos del Modelo
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    kmeans = joblib.load('artifacts/kmeans_model.pkl')
    scaler = joblib.load('artifacts/scaler.pkl')
    pca_viz = joblib.load('artifacts/pca_visualizacion.pkl')
    feature_config = joblib.load('artifacts/feature_config.pkl')
    df_data = pd.read_csv('artifacts/dataset_clusterizado.csv')
    return kmeans, scaler, pca_viz, feature_config, df_data

try:
    kmeans, scaler, pca_viz, feature_config, df_data = load_artifacts()
    num_cols = feature_config['num_cols']
    feature_names = feature_config['feature_names']
except Exception as e:
    st.error(f"Error al cargar los artefactos desde 'artifacts/': {e}")
    st.stop()

# ---------------------------------------------------------
# 3. Barra Lateral: Selección / Entrada de Datos
# ---------------------------------------------------------
st.sidebar.header("🎛️ Opciones de Entrada")
mode = st.sidebar.radio("Modo de ingreso:", ["Seleccionar Canción Existente", "Crear Nueva Canción (Custom)"])

user_num_values = {}

if mode == "Seleccionar Canción Existente":
    st.sidebar.subheader("🔍 Buscar Canción")
    df_data['song_label'] = df_data['track_name'].astype(str) + " - " + df_data['artists'].astype(str)
    selected_song_label = st.sidebar.selectbox("Selecciona una pista:", df_data['song_label'].unique())
    
    # Extraer fila seleccionada
    selected_row = df_data[df_data['song_label'] == selected_song_label].iloc[0]
    
    for col in num_cols:
        user_num_values[col] = float(selected_row[col])

else:
    st.sidebar.subheader("🎚️ Ajustar Atributos Numéricos")
    for col in num_cols:
        min_val = float(df_data[col].min())
        max_val = float(df_data[col].max())
        mean_val = float(df_data[col].mean())
        user_num_values[col] = st.sidebar.slider(col, min_value=min_val, max_value=max_val, value=mean_val)

# ---------------------------------------------------------
# 4. Procesamiento y Predicción
# ---------------------------------------------------------
# 1. Crear matriz numérica y escalarla
df_user_num = pd.DataFrame([user_num_values])[num_cols]
X_user_num_scaled = scaler.transform(df_user_num)

# 2. Reconstruir vector completo de características
df_user_scaled = pd.DataFrame(X_user_num_scaled, columns=num_cols)

# Rellenar columnas binarias/categóricas faltantes con ceros
for col in feature_names:
    if col not in df_user_scaled.columns:
        df_user_scaled[col] = 0

df_user_final = df_user_scaled[feature_names]

# 3. Predicción del Cluster y Coordenadas PCA
cluster_pred = kmeans.predict(df_user_final)[0]
pca_coords_user = pca_viz.transform(df_user_final)[0]

# ---------------------------------------------------------
# 5. Visualización de Resultados
# ---------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📌 Resultado de Clasificación")
    st.metric(label="Cluster Asignado (K-Means)", value=f"Cluster {cluster_pred}")
    
    if mode == "Seleccionar Canción Existente":
        st.write(f"**Canción:** {selected_row['track_name']}")
        st.write(f"**Artista:** {selected_row['artists']}")
        st.write(f"**Género declarado:** {selected_row['track_genre']}")
    
    st.markdown("---")
    st.subheader("📊 Valores Numéricos")
    st.dataframe(df_user_num.T.rename(columns={0: "Valor"}), use_container_width=True)

with col2:
    st.subheader("🗺️ Proyección en Espacio PCA (2D)")
    
    # Prepara dataset para gráfico de dispersión con Plotly
    if 'pca_x' not in df_data.columns:
        X_all_num = scaler.transform(df_data[num_cols])
        df_all_scaled = pd.DataFrame(X_all_num, columns=num_cols)
        for col in feature_names:
            if col not in df_all_scaled.columns:
                df_all_scaled[col] = 0
        df_all_final = df_all_scaled[feature_names]
        pca_coords_all = pca_viz.transform(df_all_final)
        df_data['pca_x'] = pca_coords_all[:, 0]
        df_data['pca_y'] = pca_coords_all[:, 1]

    # Graficar canciones existentes
    fig = px.scatter(
        df_data,
        x='pca_x',
        y='pca_y',
        color='cluster_kmeans',
        hover_data=['track_name', 'artists', 'track_genre'],
        opacity=0.5,
        title="Mapa de Clusters K-Means + Canción Seleccionada (⭐)",
        color_continuous_scale="Viridis"
    )

    # Agregar la posición de la canción del usuario como una Estrella Roja
    fig.add_trace(
        go.Scatter(
            x=[pca_coords_user[0]],
            y=[pca_coords_user[1]],
            mode='markers+text',
            marker=dict(symbol='star', size=18, color='red', line=dict(width=2, color='white')),
            text=["TU CANCIÓN"],
            textposition="top center",
            name="Canción Actual / Ingresada"
        )
    )

    fig.update_layout(
        xaxis_title="Componente Principal 1",
        yaxis_title="Componente Principal 2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 6. Recomendaciones del Mismo Cluster
# ---------------------------------------------------------
st.markdown("---")
st.subheader(f"🎧 Canciones Similares Recomendadas (Cluster {cluster_pred})")

recom_df = df_data[df_data['cluster_kmeans'] == cluster_pred].sample(
    min(5, len(df_data[df_data['cluster_kmeans'] == cluster_pred])), random_state=42
)

st.dataframe(recom_df[['track_name', 'artists', 'album_name', 'track_genre'] + num_cols[:4]], use_container_width=True)
