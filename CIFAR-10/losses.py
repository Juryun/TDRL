import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import math

class OnlineTripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector


    def forward(self, embeddings, target ,size_average=True):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        triplets = triplets.cuda()
        anchor = embeddings[triplets[:,0]]
        positive = embeddings[triplets[:,1]]
        negative = embeddings[triplets[:, 2]]

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineTripletPCALoss(nn.Module):

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletPCALoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def PCA_scale(self, x, k, center=True, scale=True):
        n, p = x.size()
        ones = torch.ones(n).view([n, 1]).cuda()
        h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
        H = torch.eye(n).cuda() - h
        X_center = torch.mm(H.double(), x.double())
        covariance = 1 / (n - 1) * torch.mm(X_center.t(), X_center).view(p, p)
        scaling = torch.sqrt(1 / torch.diag(covariance)).double() if scale else torch.ones(p).cuda().double()
        scaled_covariance = torch.mm(torch.diag(scaling).view(p, p), covariance)
        eigenvalues, eigenvectors = torch.symeig(scaled_covariance, True)
        total = eigenvalues.sum()
        if k >=1 :
            index = 1023-k
            components = (eigenvectors[:, index:])
        else :
            eigsum = 0
            index = 0
            for i in range(1024):
                eigsum = eigsum + eigenvalues[1023-i]
                if eigsum >= total*k:
                    index = 1023-i
                    break;
            components = (eigenvectors[:, index:])
        return components

    def forward(self, embeddings, target ,size_average=True):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        triplets = triplets.cuda()
        anchor = embeddings[triplets[:,0]]
        positive = embeddings[triplets[:,1]]
        negative = embeddings[triplets[:, 2]]
        data = torch.cat([anchor, positive, negative], dim=0)
        pca = self.PCA_scale(data, k=0.7)

        anchor = torch.mm(anchor, pca.float())
        positive = torch.mm(positive, pca.float())
        negative = torch.mm(negative, pca.float())

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()