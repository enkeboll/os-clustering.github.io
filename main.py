# Rerun the end-to-end pipeline with PCA-3D fallback (no external deps required).
# This reads the one-row-per-game CSV, aggregates per-user features, embeds to 3D (UMAP if available, else PCA),
# clusters with KMeans, saves artifacts, assigns clusters, and outputs a 3D PNG.

import os
import json
import joblib
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Check if UMAP is available
HAS_UMAP = False
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# ------------------------------
# Config
# ------------------------------
INPUT_EVENTS_CSV = "./learning_gameplays_500x100_with_labels.csv"  # change to your file if needed
ARTIFACT_DIR = "./cluster_artifacts_kmeans_retry"
OUTPUT_FEATURES_CSV = "./learner_features_for_clustering_kmeans_retry.csv"
OUTPUT_ASSIGNMENTS_CSV = "./learner_cluster_assignments_kmeans_retry.csv"
OUTPUT_3D_PNG = "./embedding_3d_clusters_kmeans_retry.png"
K_CLUSTERS = 6

os.makedirs(ARTIFACT_DIR, exist_ok=True)

def shannon_entropy(series: pd.Series) -> float:
    counts = series.value_counts(normalize=True)
    return float(-(counts * np.log2(np.clip(counts, 1e-12, None))).sum())

def aggregate_features(df_events: pd.DataFrame) -> pd.DataFrame:
    agg_funcs = {
        "mean_rt_ms": ["mean", "std"],
        "accuracy_rate": ["mean", "std"],
        "error_conceptual": "mean",
        "error_careless": "mean",
        "error_misread": "mean",
        "error_guessing": "mean",
        "hint_usage_rate": "mean",
        "retry_rate": "mean",
        "exploration_index": ["mean", "std"],
        "improvement_within_session": "mean",
    }
    base = df_events.groupby("user_id").agg(agg_funcs)
    base.columns = ["_".join([c for c in col if c]) for col in base.columns]
    
    # Game-type mix features
    type_means = (
        df_events.pivot_table(
            index="user_id",
            columns="game_type",
            values="accuracy_rate",
            aggfunc="mean"
        )
        .add_prefix("acc_by_type_")
        .fillna(0.0)
    )
    type_counts = df_events.groupby("user_id")["game_type"].apply(shannon_entropy).to_frame(name="game_type_entropy")
    
    # Device/input usage rates
    device_rates = (
        df_events.pivot_table(index="user_id", columns="device", values="gameplay_id", aggfunc="count")
        .fillna(0)
        .div(df_events.groupby("user_id")["gameplay_id"].count(), axis=0)
        .add_prefix("device_rate_")
    )
    input_rates = (
        df_events.pivot_table(index="user_id", columns="input_method", values="gameplay_id", aggfunc="count")
        .fillna(0)
        .div(df_events.groupby("user_id")["gameplay_id"].count(), axis=0)
        .add_prefix("input_rate_")
    )

    df_user = base.join([type_means, type_counts, device_rates, input_rates], how="left").reset_index()
    if "true_archetype" in df_events.columns:
        mode_label = df_events.groupby("user_id")["true_archetype"].agg(lambda x: x.value_counts().idxmax())
        df_user = df_user.merge(mode_label.rename("true_archetype"), left_on="user_id", right_index=True, how="left")
    return df_user

def embed_3d(X_scaled: np.ndarray, use_umap: bool = True):
    artifacts = {}
    if use_umap and HAS_UMAP:
        pca = PCA(n_components=min(20, X_scaled.shape[1], X_scaled.shape[0]), random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        umap_model = umap.UMAP(n_neighbors=20, min_dist=0.05, n_components=3, random_state=42)
        X_emb = umap_model.fit_transform(X_pca)
        artifacts["pca"] = pca
        artifacts["umap"] = umap_model
        artifacts["method"] = "umap3d"
        return X_emb, artifacts
    else:
        pca3 = PCA(n_components=3, random_state=42)
        X_emb = pca3.fit_transform(X_scaled)
        artifacts["pca3"] = pca3
        artifacts["method"] = "pca3d"
        return X_emb, artifacts

def train_pipeline(df_user: pd.DataFrame, k_clusters: int = 6):
    feature_cols = [c for c in df_user.columns if c not in ("user_id", "true_archetype")]
    X = df_user[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_emb, emb_artifacts = embed_3d(X_scaled, use_umap=True)

    kmeans = KMeans(n_clusters=k_clusters, n_init=20, random_state=42)
    labels = kmeans.fit_predict(X_emb)

    # Save artifacts
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    if emb_artifacts.get("pca"):
        joblib.dump(emb_artifacts["pca"], os.path.join(ARTIFACT_DIR, "pca.joblib"))
    if emb_artifacts.get("umap"):
        joblib.dump(emb_artifacts["umap"], os.path.join(ARTIFACT_DIR, "umap_model.joblib"))
    if emb_artifacts.get("pca3"):
        joblib.dump(emb_artifacts["pca3"], os.path.join(ARTIFACT_DIR, "pca3.joblib"))
    joblib.dump(kmeans, os.path.join(ARTIFACT_DIR, "kmeans_model.joblib"))
    joblib.dump({"method": emb_artifacts["method"]}, os.path.join(ARTIFACT_DIR, "embed_meta.joblib"))

    try:
        sil = silhouette_score(X_emb, labels)
    except Exception:
        sil = float("nan")

    return {"labels": labels, "X_emb": X_emb, "feature_cols": feature_cols, "silhouette": sil, "embed_method": emb_artifacts["method"]}

def transform_embed(X: np.ndarray):
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    meta = joblib.load(os.path.join(ARTIFACT_DIR, "embed_meta.joblib"))
    X_scaled = scaler.transform(X)

    if meta["method"] == "umap3d":
        pca = joblib.load(os.path.join(ARTIFACT_DIR, "pca.joblib"))
        umap_model = joblib.load(os.path.join(ARTIFACT_DIR, "umap_model.joblib"))
        X_pca = pca.transform(X_scaled)
        X_emb = umap_model.transform(X_pca)
    else:
        pca3 = joblib.load(os.path.join(ARTIFACT_DIR, "pca3.joblib"))
        X_emb = pca3.transform(X_scaled)
    return X_emb

def assign_clusters(df_user: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df_user.columns if c not in ("user_id", "true_archetype")]
    X = df_user[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    X_emb = transform_embed(X)
    kmeans = joblib.load(os.path.join(ARTIFACT_DIR, "kmeans_model.joblib"))
    labels = kmeans.predict(X_emb)

    out = df_user[["user_id"]].copy()
    if "true_archetype" in df_user.columns:
        out["true_archetype"] = df_user["true_archetype"].values
    out["cluster_kmeans"] = labels
    out["emb_x"] = X_emb[:, 0]
    out["emb_y"] = X_emb[:, 1]
    out["emb_z"] = X_emb[:, 2]
    return out

def plot_3d(df_assign: pd.DataFrame, path_png: str, title: str):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_assign["emb_x"].values, df_assign["emb_y"].values, df_assign["emb_z"].values,
               c=df_assign["cluster_kmeans"].values, s=25)
    ax.set_title(title)
    ax.set_xlabel("Dim-1")
    ax.set_ylabel("Dim-2")
    ax.set_zlabel("Dim-3")
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close(fig)

# Execute
events = pd.read_csv(INPUT_EVENTS_CSV)
features = aggregate_features(events)
features.to_csv(OUTPUT_FEATURES_CSV, index=False)

train_info = train_pipeline(features, k_clusters=K_CLUSTERS)

assignments = assign_clusters(features)
assignments.to_csv(OUTPUT_ASSIGNMENTS_CSV, index=False)

title = f"3D Embedding ({train_info['embed_method']}) - KMeans Clusters"
plot_3d(assignments, OUTPUT_3D_PNG, title)

{
    "artifacts_dir": ARTIFACT_DIR,
    "features_csv": OUTPUT_FEATURES_CSV,
    "assignments_csv": OUTPUT_ASSIGNMENTS_CSV,
    "embedding_3d_png": OUTPUT_3D_PNG,
    "silhouette_on_embedding": float(train_info["silhouette"]),
    "clusters_count": int(len(np.unique(assignments['cluster_kmeans']))),
    "embed_method": train_info["embed_method"],
}
