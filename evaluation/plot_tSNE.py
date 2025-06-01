import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def plot_tsne(embeddings, labels, save_path, title=None, n_jobs=2, metric="euclidean"):
    tsne = TSNE(n_components=2, random_state=42, init='pca', n_jobs=n_jobs, metric=metric)
    emb_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter, label='Label')
    plt.title(title if title else "t-SNE Plot")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def l2_normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1
    return embeddings / norms


def create_plots(add_l2_norm=True):
    RESULTS_DIR = Path('./results/models')
    SPLITS = ['train', 'val', 'test']

    for model_name in os.listdir(RESULTS_DIR):
        model_dir = RESULTS_DIR / model_name
        if not model_dir.is_dir():
            continue
        print(f"Processing model: {model_name}")
        for split in SPLITS:
            emb_path = model_dir / f"{split}_embeddings.pt"
            if not emb_path.exists():
                print(f"  - {split} embeddings not found, skipping.")
                continue
            print(f"  - Creating t-SNE for {split} embeddings...")
            data = torch.load(emb_path, map_location='cpu')
            embeddings = data['embeddings']
            labels = data['labels']
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()

            save_path = model_dir / f"{split}_tsne.png"
            plot_tsne(embeddings, labels, save_path, title=f"{model_name} - {split}", metric="euclidean")
            print(f"    Saved t-SNE plot to {save_path}")

            if add_l2_norm:
                embeddings_norm = l2_normalize(embeddings)
                save_path_norm = model_dir / f"{split}_tsne_cosine.png"
                plot_tsne(embeddings_norm, labels, save_path_norm, title=f"{model_name} - {split} (cosine)", metric="cosine")
                print(f"    Saved t-SNE plot to {save_path_norm}")
