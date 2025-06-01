# interpolation_analysis.py

import torch
import torch.nn.functional as F
import os
import pandas as pd
from pathlib import Path
from itertools import combinations
from collections import defaultdict

@torch.no_grad()
def interpolate_analysis(models_dict, n_samples=5, n_interp=10, device='cpu'):
    DATA_DIR = Path('./results/models')
    SPLITS = ['train', 'val', 'test']
    RESULT_CSV = './results/interpolation_precision.csv'

    rows = []

    for model_idx, (model_name, model_obj) in enumerate(models_dict.items()):
        model_dir = DATA_DIR / model_name
        model_path = model_dir / "model.pt"
        if not model_dir.is_dir() or not model_path.exists():
            print(f"[{model_idx + 1}] Skipping {model_name}: directory or model file missing.")
            continue

        print(f"[{model_idx + 1}] Processing model: {model_name}")

        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model_obj.load_state_dict(checkpoint['model_state_dict'])
        model = model_obj.to(device).eval()

        for split in SPLITS:
            emb_path = model_dir / f"{split}_embeddings.pt"
            if not emb_path.exists():
                print(f"  - {split} embeddings not found, skipping.")
                continue

            print(f"  - Evaluating {split} embeddings...")

            data = torch.load(emb_path, map_location=device)
            embeddings = data['embeddings'].to(device)
            labels = data['labels'].to(device)
            num_classes = checkpoint['num_classes']

            # Get correctly classified samples
            correct_preds = []
            for i in range(0, len(embeddings), 100):
                batch_embeds = embeddings[i:i+100]
                if hasattr(model, 'use_dropout') and model.use_dropout:
                    logits = model.clas(model.dropout(batch_embeds))
                else:
                    logits = model.clas(batch_embeds)
                preds = torch.argmax(logits, dim=1)
                correct = (preds == labels[i:i+100])
                correct_preds.extend(correct.cpu().tolist())

            correct_preds = torch.tensor(correct_preds, dtype=torch.bool)
            embeddings = embeddings[correct_preds]
            labels = labels[correct_preds]

            class_precisions = {}
            for cls in range(num_classes):
                cls_indices = (labels == cls).nonzero(as_tuple=True)[0]
                if len(cls_indices) < n_samples:
                    continue

                selected = cls_indices[:n_samples]
                cls_embeddings = embeddings[selected]

                interpolated_correct = 0
                total_interp = 0

                for i, j in combinations(range(n_samples), 2):
                    e1 = cls_embeddings[i]
                    e2 = cls_embeddings[j]
                    for alpha in torch.linspace(0, 1, steps=n_interp):
                        interp = (1 - alpha) * e1 + alpha * e2
                        interp = interp.unsqueeze(0)
                        if hasattr(model, 'use_dropout') and model.use_dropout:
                            logits = model.clas(model.dropout(interp))
                        else:
                            logits = model.clas(interp)
                        pred = logits.argmax(dim=1).item()
                        if pred == cls:
                            interpolated_correct += 1
                        total_interp += 1

                precision = interpolated_correct / total_interp if total_interp > 0 else 0.0
                class_precisions[f'class_{cls}_precision'] = precision

            row = {
                'model': model_name,
                'split': split,
                **class_precisions
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    Path('./results').mkdir(exist_ok=True)
    df.to_csv(RESULT_CSV, index=False)
    print(f"Saved interpolation precision results to {RESULT_CSV}")
