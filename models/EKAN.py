import torch.nn as nn
import torch.nn.functional as F
from models.feature_extractor import FeatureExtractor

import sys
import os
KAN_SRC_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'kan_repos', 'efficient-kan', 'src')
)
if KAN_SRC_PATH not in sys.path:
    sys.path.insert(0, KAN_SRC_PATH)

from efficient_kan import kan as EffKan

class EKANModel(nn.Module):
    def __init__(self, feature_extractor=None,
                 emb_dim=256, n_class=10,
                 use_dropout=False, dropout_rate=0.5,
                 KAN_params=None):
        super(EKANModel, self).__init__()

        if feature_extractor is None:
            feature_extractor = FeatureExtractor()
        self.extr = feature_extractor

        # self.emb = nn.Linear(feature_extractor.n_out, emb_dim)
        if KAN_params is None:
            KAN_params = {}
        self.emb = EffKan.KANLinear(feature_extractor.n_out,
                                    emb_dim,
                                    **KAN_params)

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        self.clas = nn.Linear(emb_dim, n_class)
    

    def forward(self, x, return_embeddings=False):
        x = self.extr(x)
        x = self.emb(x)

        if return_embeddings:
            return x
        
        if self.use_dropout:
            x = self.dropout(x)

        x = self.clas(x)
        return x