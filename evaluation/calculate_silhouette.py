import numpy as np
import os
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import silhouette_score

def evaluate_silhouette_scores(models_path=None):
    if models_path is None:
        models_path = './results/models'

    RESULTS_DIR = Path(models_path)
    SPLITS = ['train', 'val', 'test']

    rows = []

    for model_idx, model_name in enumerate(os.listdir(RESULTS_DIR)):
        model_dir = RESULTS_DIR / model_name
        if not model_dir.is_dir():
            continue
        print(f"[{model_idx+1}] Processing model: {model_name}")
        for split in SPLITS:
            emb_path = model_dir / f"{split}_embeddings.pt"
            if not emb_path.exists():
                print(f"  - {split} embeddings not found, skipping.")
                continue
            print(f"  - Evaluating {split} embeddings...")
            data = torch.load(emb_path, map_location='cpu')
            embeddings = data['embeddings']
            labels = data['labels']

            # Move to CPU and convert to numpy
            if isinstance(embeddings, torch.Tensor):
                embeddings_np = embeddings.to('cpu').numpy()
            else:
                embeddings_np = embeddings
            if isinstance(labels, torch.Tensor):
                labels_np = labels.to('cpu').numpy()
            else:
                labels_np = labels

            # Silhouette score for cosine and euclidean
            try:
                sil_euclidean = silhouette_score(embeddings_np, labels_np, metric='euclidean')
            except Exception as e:
                print(f"    - Silhouette (euclidean) failed: {e}")
                sil_euclidean = np.nan
            try:
                sil_cosine = silhouette_score(embeddings_np, labels_np, metric='cosine')
            except Exception as e:
                print(f"    - Silhouette (cosine) failed: {e}")
                sil_cosine = np.nan

            row = {
                'model': model_name,
                'split': split,
                'silhouette_euclidean': sil_euclidean,
                'silhouette_cosine': sil_cosine,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv('./results/silhouette_scores_all.csv', index=False)
    print("Saved silhouette scores to ./results/silhouette_scores_all.csv")


if __name__ == "__main__":
    evaluate_silhouette_scores()