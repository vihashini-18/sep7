# app.py
import io
import math
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# ---------- CONFIG ----------
DATA_PATH = "Wholesale customers data.csv"   # put your CSV in same folder
USERNAME = "admin"
PASSWORD = "123"
N_CLUSTERS = 3  # change if you prefer a different number
# ----------------------------

# ---------- Helper functions ----------
def safe_load_data(path):
    try:
        df = pd.read_csv(path)
        # ensure required columns exist
        required = ['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']
        if not all(col in df.columns for col in required):
            raise ValueError(f"CSV must contain columns: {required}")
        return df
    except Exception as e:
        # fallback: tiny random dataset so app still runs
        print(f"Warning loading CSV ({e}); using random sample dataset instead.")
        rng = np.random.default_rng(42)
        df = pd.DataFrame(rng.integers(100, 20000, size=(60,6)), columns=['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen'])
        return df

def compute_centroid_dict(X_scaled, labels):
    """Return dict: label -> centroid (mean vector). Ignore noise label -1."""
    centroids = {}
    unique = np.unique(labels)
    for lbl in unique:
        if lbl == -1:
            continue
        mask = labels == lbl
        if mask.sum() > 0:
            centroids[int(lbl)] = X_scaled[mask].mean(axis=0)
    return centroids

def assign_by_nearest_centroid(x_scaled, centroids):
    """Assign x_scaled (1xD) to nearest centroid. centroids: {label: vector}"""
    if len(centroids) == 0:
        return None
    best_lbl = None
    best_dist = math.inf
    for lbl, c in centroids.items():
        d = np.linalg.norm(x_scaled.ravel() - c)
        if d < best_dist:
            best_dist = d
            best_lbl = int(lbl)
    return best_lbl

def create_combined_plot(X_pca, labels_dict, x_input_pca, pred_labels):
    """
    labels_dict: {'KMeans': labels_k, 'Hierarchical': labels_h, 'DBSCAN': labels_d}
    x_input_pca: (1,2)
    pred_labels: dict of predicted labels to annotate
    returns PIL.Image
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    algos = ['KMeans', 'Hierarchical', 'DBSCAN']
    cmap = plt.cm.get_cmap("tab10")
    for i, algo in enumerate(algos):
        ax = axes[i]
        lbls = labels_dict[algo]
        unique = np.unique(lbls)
        # map noise (-1) to a color index optionally
        for j, u in enumerate(unique):
            mask = lbls == u
            color = cmap(j % 10) if u != -1 else (0.6,0.6,0.6)  # gray for noise
            ax.scatter(X_pca[mask,0], X_pca[mask,1], c=[color], s=40, label=f"Cluster {u}", alpha=0.6, edgecolors='k', linewidths=0.2)
        # plot input
        ax.scatter(x_input_pca[0,0], x_input_pca[0,1], c='red', marker='X', s=200, label='Your Input', edgecolors='k')
        ax.set_title(f"{algo} (pred: {pred_labels.get(algo,'N/A')})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ---------- Load & train (once at startup) ----------
df = safe_load_data(DATA_PATH)
features = ['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']
X = df[features].values.astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans (can predict on new data)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Hierarchical (no predict method) - we compute labels on training data
hier_labels = AgglomerativeClustering(n_clusters=N_CLUSTERS).fit_predict(X_scaled)

# DBSCAN
db_labels = DBSCAN(eps=1.5, min_samples=3).fit_predict(X_scaled)

# Precompute centroids/means for hierarchical and dbscan (for assignment)
hier_centroids = compute_centroid_dict(X_scaled, hier_labels)
db_centroids = compute_centroid_dict(X_scaled, db_labels)

# Prepare PCA for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

labels_dict_train = {
    'KMeans': kmeans_labels,
    'Hierarchical': hier_labels,
    'DBSCAN': db_labels
}

# ---------- Gradio app ----------
def login_action(username, password):
    if username == USERNAME and password == PASSWORD:
        return gr.update(visible=True), "✅ Login successful! Enter data below."
    else:
        return gr.update(visible=False), "❌ Invalid username/password"

def predict_and_visualize(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen):
    try:
        user_arr = np.array([[Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen]], dtype=float)
        user_scaled = scaler.transform(user_arr)  # shape (1, D)

        # KMeans predict
        km_label = int(kmeans.predict(user_scaled)[0])

        # Assign hierarchical via nearest centroid (mean)
        hier_label = assign_by_nearest_centroid(user_scaled, hier_centroids)
        # Assign DBSCAN similarly
        db_label = assign_by_nearest_centroid(user_scaled, db_centroids)
        if db_label is None:
            db_label = -1  # no DBSCAN clusters (all noise?) -> mark -1

        # PCA transform input for plotting
        user_pca = pca.transform(user_scaled)  # shape (1,2)

        pred_labels = {'KMeans': km_label, 'Hierarchical': hier_label, 'DBSCAN': db_label}
        img = create_combined_plot(X_pca, labels_dict_train, user_pca, pred_labels)

        return f"{km_label}", f"{hier_label}", f"{db_label}", img

    except Exception as e:
        # return error strings and no image
        msg = f"Error: {str(e)}"
        return msg, msg, msg, None

# Build Blocks UI with login and clustering (clustering hidden until login)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔐 Wholesale Customers — Clustering Explorer")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Login")
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_msg = gr.Markdown("")  # text for messages

        with gr.Column(scale=2):
            gr.Markdown("**Instructions**: Login first (default: `admin` / `123`). Then enter the 6 numeric features and click *Predict*.")
    # clustering container: hidden until login
    with gr.Column(visible=False) as cluster_col:
        gr.Markdown("### 🧪 Enter customer features")
        with gr.Row():
            fresh_in = gr.Number(label="Fresh")
            milk_in = gr.Number(label="Milk")
            grocery_in = gr.Number(label="Grocery")
        with gr.Row():
            frozen_in = gr.Number(label="Frozen")
            det_in = gr.Number(label="Detergents_Paper")
            deli_in = gr.Number(label="Delicassen")
        predict_btn = gr.Button("Predict Clusters")
        with gr.Row():
            km_out = gr.Textbox(label="KMeans Cluster", interactive=False)
            hier_out = gr.Textbox(label="Hierarchical Cluster (nearest mean)", interactive=False)
            db_out = gr.Textbox(label="DBSCAN Cluster (nearest mean)", interactive=False)
        plot_out = gr.Image(label="Cluster visualization (PCA 2D)")

    # Wire login and predict
    login_btn.click(fn=login_action, inputs=[username, password], outputs=[cluster_col, login_msg])
    predict_btn.click(fn=predict_and_visualize,
                      inputs=[fresh_in, milk_in, grocery_in, frozen_in, det_in, deli_in],
                      outputs=[km_out, hier_out, db_out, plot_out])

# Launch
if __name__ == "__main__":
    demo.launch()
