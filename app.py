import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import pairwise_distances

# =========================================================
# 1. CONFIGURACIÓN DE LA APP
# =========================================================
st.set_page_config(
    page_title="Spotify Vibe Explorer",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Spotify Vibe Explorer & Cluster Predictor")
st.markdown(
    """
    Analiza canciones existentes o crea una canción manualmente.
    El sistema aplica el mismo preprocesamiento utilizado durante el entrenamiento,
    asigna uno de los **5 clusters K-Means**, muestra su posición en PCA y recomienda
    canciones cercanas dentro del mismo perfil.
    """
)

# =========================================================
# 2. CARGA DE ARTEFACTOS
# =========================================================
@st.cache_resource
def load_artifacts():
    model_path = (
        "artifacts/kmeans_model.pkl"
        if os.path.exists("artifacts/kmeans_model.pkl")
        else "artifacts/best_model.pkl"
    )

    kmeans = joblib.load(model_path)
    scaler = joblib.load("artifacts/scaler.pkl")
    pca_viz = joblib.load("artifacts/pca_visualizacion.pkl")
    feature_config = joblib.load("artifacts/feature_config.pkl")
    df_data = pd.read_csv("artifacts/dataset_clusterizado.csv")

    return kmeans, scaler, pca_viz, feature_config, df_data


try:
    kmeans, scaler, pca_viz, feature_config, df_data = load_artifacts()
except Exception as e:
    st.error(f"Error al cargar los artefactos del modelo: {e}")
    st.stop()

num_cols = feature_config.get("num_cols", [])
bin_cols = feature_config.get("bin_cols", ["explicit", "mode"])
cat_cols = feature_config.get("cat_cols", ["key", "time_signature"])
feature_names = feature_config["feature_names"]

# =========================================================
# 3. FUNCIONES AUXILIARES
# =========================================================
def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def build_feature_vector(values):
    """
    Reconstruye exactamente el vector de características usado por K-Means:
    - escala variables numéricas con StandardScaler
    - mantiene binarias
    - reconstruye one-hot para key y time_signature
    """
    raw_num = pd.DataFrame(
        [[safe_float(values.get(c, 0.0)) for c in num_cols]],
        columns=num_cols
    )

    scaled_num = pd.DataFrame(
        scaler.transform(raw_num),
        columns=num_cols
    )

    row = {name: 0.0 for name in feature_names}

    # Numéricas escaladas
    for c in num_cols:
        if c in row:
            row[c] = float(scaled_num.loc[0, c])

    # Binarias
    for c in bin_cols:
        if c in row:
            row[c] = safe_float(values.get(c, 0))

    # Categóricas one-hot
    for c in cat_cols:
        value = values.get(c, None)
        if value is None:
            continue

        candidates = [f"{c}_{value}"]

        try:
            fv = float(value)
            candidates.extend([
                f"{c}_{int(fv)}",
                f"{c}_{fv}"
            ])
        except Exception:
            pass

        for candidate in candidates:
            if candidate in row:
                row[candidate] = 1.0
                break

    return pd.DataFrame([row], columns=feature_names)


@st.cache_data
def prepare_dataset_features(df):
    rows = []
    for _, r in df.iterrows():
        rows.append(
            build_feature_vector(r.to_dict()).iloc[0].to_numpy(dtype=float)
        )
    return np.vstack(rows)


def level_from_z(z):
    if z >= 0.65:
        return "Alta"
    if z <= -0.65:
        return "Baja"
    return "Media"


def cluster_profile(cluster_id):
    """
    Genera un nombre interpretativo a partir del centroide.
    No usa el género como entrada ni inventa una etiqueta externa.
    """
    center = np.asarray(kmeans.cluster_centers_[cluster_id], dtype=float)

    interesting = [
        c for c in [
            "danceability",
            "energy",
            "acousticness",
            "instrumentalness",
            "valence",
            "speechiness",
            "liveness",
            "tempo",
            "loudness"
        ]
        if c in feature_names
    ]

    vals = {c: center[feature_names.index(c)] for c in interesting}

    tags = []

    if vals.get("acousticness", 0) > 0.55:
        tags.append("acústico")
    if vals.get("energy", 0) > 0.55:
        tags.append("energético")
    if vals.get("danceability", 0) > 0.55:
        tags.append("bailable")
    if vals.get("instrumentalness", 0) > 0.55:
        tags.append("instrumental")
    if vals.get("valence", 0) > 0.55:
        tags.append("positivo")
    if vals.get("speechiness", 0) > 0.55:
        tags.append("vocal/hablado")

    # Añadir rasgos bajos útiles cuando ayuden a describir el perfil
    if vals.get("energy", 0) < -0.55 and "energético" not in tags:
        tags.append("de baja energía")
    if vals.get("acousticness", 0) < -0.55 and "acústico" not in tags:
        tags.append("poco acústico")

    if not tags:
        ranked = sorted(
            vals.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:2]

        for feat, val in ranked:
            if val >= 0:
                tags.append(feat.replace("_", " "))
            else:
                tags.append(f"bajo {feat.replace('_', ' ')}")

    profile_name = " y ".join(tags[:2]).capitalize()
    levels = {c: level_from_z(vals[c]) for c in vals}

    return profile_name, levels, vals


def generate_interpretation(profile_levels):
    """
    Construye una explicación textual simple y entendible
    usando solo niveles del centroide del cluster.
    """
    parts = []

    if profile_levels.get("acousticness") == "Alta":
        parts.append("alta presencia acústica")
    elif profile_levels.get("acousticness") == "Baja":
        parts.append("baja presencia acústica")

    if profile_levels.get("energy") == "Alta":
        parts.append("energía elevada")
    elif profile_levels.get("energy") == "Baja":
        parts.append("energía reducida")

    if profile_levels.get("danceability") == "Alta":
        parts.append("alta facilidad para el baile")

    if profile_levels.get("instrumentalness") == "Alta":
        parts.append("fuerte componente instrumental")

    if profile_levels.get("valence") == "Alta":
        parts.append("tendencia a una sonoridad más positiva")

    if profile_levels.get("speechiness") == "Alta":
        parts.append("presencia importante de contenido hablado")

    if not parts:
        return (
            "El perfil del cluster presenta valores cercanos al promedio "
            "en varias de sus principales características de audio."
        )

    if len(parts) == 1:
        return f"Este cluster se caracteriza principalmente por {parts[0]}."

    return (
        "Este cluster se caracteriza principalmente por "
        + ", ".join(parts[:-1])
        + " y "
        + parts[-1]
        + "."
    )


# =========================================================
# 4. PREPARAR CATÁLOGO
# =========================================================
with st.spinner("Preparando el catálogo para clasificación y similitud..."):
    X_all = prepare_dataset_features(df_data)
    predicted_all = kmeans.predict(X_all)

df_data = df_data.copy()
df_data["cluster_app"] = predicted_all

if "pca_x" not in df_data.columns or "pca_y" not in df_data.columns:
    coords = pca_viz.transform(X_all)
    df_data["pca_x"] = coords[:, 0]
    df_data["pca_y"] = coords[:, 1]

# =========================================================
# 5. ENTRADA DEL USUARIO
# =========================================================
st.sidebar.header("🎛️ Opciones de entrada")

mode = st.sidebar.radio(
    "Modo de ingreso:",
    [
        "Seleccionar canción existente",
        "Crear nueva canción manualmente"
    ]
)

values = {}
song_name = "Canción personalizada"
artist_name = "Usuario"
selected_index = None
selected_row = None

if mode == "Seleccionar canción existente":
    st.sidebar.subheader("🔎 Buscar canción")

    labels = (
        df_data["track_name"].fillna("Sin título").astype(str)
        + " — "
        + df_data["artists"].fillna("Artista desconocido").astype(str)
    )

    selected_index = st.sidebar.selectbox(
        "Selecciona una pista:",
        options=df_data.index,
        format_func=lambda i: labels.loc[i]
    )

    selected_row = df_data.loc[selected_index]
    values = selected_row.to_dict()

    song_name = str(selected_row.get("track_name", "Sin título"))
    artist_name = str(selected_row.get("artists", "Artista desconocido"))

else:
    st.sidebar.subheader("✍️ Información")
    song_name = st.sidebar.text_input("Nombre de la canción", "Mi canción")
    artist_name = st.sidebar.text_input("Artista", "Artista nuevo")

    st.sidebar.subheader("🎚️ Características de audio")

    for col in num_cols:
        min_val = safe_float(df_data[col].min())
        max_val = safe_float(df_data[col].max())
        mean_val = safe_float(df_data[col].mean())

        span = max_val - min_val
        step = max(span / 200.0, 0.001)

        values[col] = st.sidebar.slider(
            col,
            min_value=min_val,
            max_value=max_val,
            value=min(max(mean_val, min_val), max_val),
            step=step
        )

    if "explicit" in bin_cols:
        values["explicit"] = int(
            st.sidebar.checkbox("Explicit", value=False)
        )

    if "mode" in bin_cols:
        values["mode"] = st.sidebar.selectbox(
            "Mode",
            options=[0, 1],
            format_func=lambda x: "Menor (0)" if x == 0 else "Mayor (1)"
        )

    if "key" in cat_cols and "key" in df_data.columns:
        keys = sorted(df_data["key"].dropna().unique().tolist())
        values["key"] = st.sidebar.selectbox("Key", keys)

    if "time_signature" in cat_cols and "time_signature" in df_data.columns:
        signatures = sorted(
            df_data["time_signature"].dropna().unique().tolist()
        )
        values["time_signature"] = st.sidebar.selectbox(
            "Time signature",
            signatures
        )

# =========================================================
# 6. PREDICCIÓN
# =========================================================
X_user = build_feature_vector(values)

cluster_pred = int(kmeans.predict(X_user)[0])
pca_user = pca_viz.transform(X_user)[0]

profile_name, profile_levels, profile_vals = cluster_profile(cluster_pred)
interpretation = generate_interpretation(profile_levels)

center = np.asarray(
    kmeans.cluster_centers_[cluster_pred],
    dtype=float
).reshape(1, -1)

dist_center = float(pairwise_distances(X_user, center)[0, 0])

mask_cluster = df_data["cluster_app"].to_numpy() == cluster_pred

cluster_distances_to_center = pairwise_distances(
    X_all[mask_cluster],
    center
).ravel()

# Percentil inverso:
# valor alto => más cerca del centro que gran parte del cluster
relative_position = 100.0 * np.mean(
    cluster_distances_to_center >= dist_center
)

relative_position = float(
    np.clip(relative_position, 0, 100)
)

# =========================================================
# 7. RESULTADO PRINCIPAL
# =========================================================
st.markdown("---")

left, right = st.columns([1, 2])

with left:
    st.subheader("🎯 Resultado del análisis")
    st.caption(f"{song_name} — {artist_name}")

    metric1, metric2 = st.columns(2)

    metric1.metric(
        "Cluster asignado",
        f"Cluster {cluster_pred}"
    )

    metric2.metric(
        "Posición relativa dentro del cluster",
        f"{relative_position:.1f}%"
    )

    st.markdown(f"### 🎵 Perfil: {profile_name}")

    st.write(interpretation)

    if selected_row is not None:
        genre = selected_row.get("track_genre", "No disponible")
        st.write(f"**Género declarado:** {genre}")

    # Mostrar solo las características más relevantes
    preferred_order = [
        "energy",
        "danceability",
        "acousticness",
        "instrumentalness",
        "valence"
    ]

    key_rows = []

    for feat in preferred_order:
        if feat in profile_levels:
            key_rows.append({
                "Característica": feat,
                "Nivel del cluster": profile_levels[feat]
            })

    if key_rows:
        st.markdown("#### Características destacadas")
        st.dataframe(
            pd.DataFrame(key_rows),
            hide_index=True,
            use_container_width=True
        )

    st.caption(
        "La posición relativa indica qué tan cerca se encuentra la canción "
        "del centroide respecto a las demás canciones del mismo cluster. "
        "No representa una probabilidad."
    )

with right:
    st.subheader("🗺️ Proyección PCA (2D)")

    fig = px.scatter(
        df_data,
        x="pca_x",
        y="pca_y",
        color="cluster_app",
        hover_data=[
            "track_name",
            "artists",
            "track_genre"
        ],
        opacity=0.42,
        title="Mapa de clusters K-Means + canción analizada",
        color_continuous_scale="Viridis"
    )

    fig.add_trace(
        go.Scatter(
            x=[pca_user[0]],
            y=[pca_user[1]],
            mode="markers+text",
            marker=dict(
                symbol="star",
                size=24,
                color="red",
                line=dict(
                    width=2,
                    color="white"
                )
            ),
            text=[song_name],
            textposition="top center",
            textfont=dict(size=12),
            name="Canción analizada"
        )
    )

    fig.update_layout(
        xaxis_title="Componente Principal 1",
        yaxis_title="Componente Principal 2",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# 8. DETALLE DE VARIABLES
# =========================================================
with st.expander("📊 Ver todas las características utilizadas"):
    display_values = {
        c: values.get(c, None)
        for c in num_cols + bin_cols + cat_cols
        if c in values
    }

    st.dataframe(
        pd.DataFrame({
            "Característica": display_values.keys(),
            "Valor": display_values.values()
        }),
        hide_index=True,
        use_container_width=True
    )

# =========================================================
# 9. RECOMENDACIONES
# =========================================================
st.markdown("---")
st.subheader(
    f"🎧 Canciones más similares dentro del Cluster {cluster_pred}"
)

cluster_indices = np.where(mask_cluster)[0]
cluster_X = X_all[cluster_indices]

dists = pairwise_distances(
    X_user,
    cluster_X
).ravel()

order = np.argsort(dists)

recommended_positions = []

for pos in order:
    original_pos = cluster_indices[pos]
    original_index = df_data.index[original_pos]

    if (
        selected_index is not None
        and original_index == selected_index
    ):
        continue

    recommended_positions.append(
        (original_pos, dists[pos])
    )

    if len(recommended_positions) == 5:
        break

if recommended_positions:
    # Escala relativa para interpretación visual.
    max_reference = max(
        float(np.percentile(dists, 90)),
        1e-9
    )

    rows = []

    for original_pos, distance in recommended_positions:
        r = df_data.iloc[original_pos]

        similarity = 100 * max(
            0.0,
            1.0 - float(distance) / max_reference
        )

        rows.append({
            "Canción": r.get("track_name", ""),
            "Artista": r.get("artists", ""),
            "Álbum": r.get("album_name", ""),
            "Género": r.get("track_genre", ""),
            "Similitud relativa": f"{similarity:.1f}%"
        })

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Las recomendaciones se ordenan por distancia en el mismo espacio "
        "de características utilizado por K-Means. La similitud mostrada es "
        "una medida relativa para facilitar la interpretación."
    )

else:
    st.info(
        "No se encontraron otras canciones disponibles "
        "para recomendar en este cluster."
    )

# =========================================================
# 10. NOTA FINAL
# =========================================================
st.markdown("---")

st.caption(
    "PCA se utiliza únicamente para visualización en dos dimensiones. "
    "La asignación del cluster se realiza con el vector completo de "
    "características del modelo."
)
