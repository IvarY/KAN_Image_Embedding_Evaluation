import torch.nn as nn
import torch.nn.functional as F
from models.feature_extractor import FeatureExtractor

class MLPModel(nn.Module):
    def __init__(self, feature_extractor=FeatureExtractor(), emb_dim=256, n_class=10, use_dropout=True, dropout_rate=0.5):
        super(MLPModel, self).__init__()

        self.extr = feature_extractor

        self.emb = nn.Linear(feature_extractor.n_out, emb_dim)

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        self.clas = nn.Linear(emb_dim, n_class)
    

    def forward(self, x, return_embeddings=False):
        x = self.extr(x)
        x = self.emb(x)

        if return_embeddings:
            return x
        
        x = F.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        
        x = self.clas(x)
        return x