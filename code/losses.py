import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses
from triplet_pca_margin_loss import TripletPCAMarginLoss

class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss

class TripletPCALoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletPCALoss, self).__init__()
        self.margin = margin
        self.k = 0.75
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets='semihard')
        self.loss_func = TripletPCAMarginLoss(margin=self.margin)

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss