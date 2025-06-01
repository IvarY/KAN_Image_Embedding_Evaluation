import numpy as np
from sklearn.metrics import pairwise_distances
import os
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path


def evaluate_embeddings(use_torch=True, device='cpu'):
    RESULTS_DIR = Path('./results/models')
    SPLITS = ['train', 'val', 'test']
    K_LIST = [1, 5, 10]

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

            embeddings = embeddings.to(device)
            labels = labels.to(device)
            precisions_cosine = knn_precision_gpu(embeddings, labels, k_list=K_LIST, metric='cosine')
            precisions_euclidean = knn_precision_gpu(embeddings, labels, k_list=K_LIST, metric='euclidean')
            
            row = {
                'model': model_name,
                'split': split,
            }
            for k, prec in zip(K_LIST, precisions_cosine):
                row[f'knn_precision_cosine_{k}'] = prec
            for k, prec in zip(K_LIST, precisions_euclidean):
                row[f'knn_precision_euclidean_{k}'] = prec
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv('./results/knn_precision_all.csv', index=False)
    print("Saved KNN precision results to ./results/knn_precision_all.csv")


def knn_precision_gpu(embeddings, labels, k_list=[1, 5, 10], metric='cosine'):
    if metric == 'cosine':
        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # Compute cosine similarity matrix
        sim_matrix = embeddings @ embeddings.T
        # Set diagonal to -inf to exclude self in topk
        sim_matrix.fill_diagonal_(-float('inf'))
        # Get top k indices with highest similarity
        max_k = max(k_list)
        knn_idx = sim_matrix.topk(max_k, dim=1).indices
    elif metric == 'euclidean':
        # Compute pairwise Euclidean distances
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        # Set diagonal to +inf to exclude self in topk
        dist_matrix.fill_diagonal_(float('inf'))
        max_k = max(k_list)
        # Get top k indices with smallest distances
        knn_idx = dist_matrix.topk(max_k, dim=1, largest=False).indices
    else:
        raise ValueError(f"Unsupported metric '{metric}', choose 'cosine' or 'euclidean'.")

    knn_labels = labels[knn_idx]
    matches = (knn_labels == labels.unsqueeze(1))

    results = []
    for k in k_list:
        precision = matches[:, :k].sum(dim=1).float().mean().item() / k
        results.append(precision)

    return results


# def knn_precision(embeddings, labels, k_list=[1, 5, 10], metric='cosine'):

#     if isinstance(embeddings, torch.Tensor):
#         embeddings = embeddings.cpu().numpy()
#     if isinstance(labels, torch.Tensor):
#         labels = labels.cpu()

#     max_k = max(k_list)
#     distances = pairwise_distances(embeddings, embeddings, metric=metric)
#     np.fill_diagonal(distances, float('inf'))

#     knn_idx = distances.argsort(axis=1)[:, :max_k]
#     knn_labels = labels[knn_idx]

#     results = []
#     for k in k_list:
#         topk_labels = knn_labels[:, :k]
#         matches = (topk_labels == labels.unsqueeze(1)).sum(dim=1)
#         precision = (matches.float() / k).mean().item()
#         results.append(precision)

#     return results