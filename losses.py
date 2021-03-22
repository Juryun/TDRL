import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import math
from visdom import Visdom

count = 1
avg = torch.ones(512,512,requires_grad=True)
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative,size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class SymmTripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(SymmTripletLoss, self).__init__()
        self.margin = margin

    def get_symmetric_point(self, anchor, pos):
        P = pos
        A = anchor  # basis
        A_norm = torch.norm(A)
        U = A / A_norm
        R = A + torch.sum(torch.mul((P - A), U), dim=1, keepdim=True) * U
        Q = 1 * (2 * R - P)
        Q = Q / torch.norm(Q) * A_norm
        return Q

    def forward(self, anchor, positive, negative, target, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)

        sym1 = self.get_symmetric_point(anchor, positive)
        sym2 = self.get_symmetric_point(positive, anchor)

        pts_list = []
        pts_list.append(anchor)
        pts_list.append(positive)
        pts_list.append(sym1)
        pts_list.append(sym2)

        negative_distance_list = []

        for pts in pts_list:
            distance = (pts - negative).pow(2).sum(1)
            negative_distance_list.append(distance)

        list0 = negative_distance_list[0].reshape(1, negative_distance_list[0].shape[0]).transpose(0,1)
        list1 = negative_distance_list[1].reshape(1, negative_distance_list[1].shape[0]).transpose(0,1)
        list2 = negative_distance_list[2].reshape(1, negative_distance_list[2].shape[0]).transpose(0, 1)
        list3 = negative_distance_list[3].reshape(1, negative_distance_list[3].shape[0]).transpose(0, 1)
        dis_array = torch.cat((list0, list1, list2, list3), dim=1)
        min_array = torch.min(dis_array, dim=1)

        distance_negative = min_array.values
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class OnlineTripletPCALoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

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
        gap = distance_positive - distance_negative
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class TripletPCALoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletPCALoss, self).__init__()
        self.margin = margin

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
        eigsum = 0
        index = 0
        for i in range(1024):
            eigsum = eigsum + eigenvalues[1023-i]
            if eigsum >= total*k:
                index = 1023-i
                break;
        components = (eigenvectors[:, index:])
        return components

    def forward(self, anchor, positive, negative, size_average=True):
        data = torch.cat([anchor, positive, negative], dim=0)
        pca = self.PCA_scale(data, k=0.7)

        anchor = torch.mm(anchor, pca.float())
        positive = torch.mm(positive, pca.float())
        negative = torch.mm(negative, pca.float())

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)
        gap = distance_positive - distance_negative
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class TripletPCALossPlus(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletPCALossPlus, self).__init__()
        self.margin = margin

    def findCos(self, ev1):
        basis = torch.zeros(1,512)
        basis[0][2] = 1
        basis = basis.cuda()
        basisnorm = torch.norm(basis)
        basis = basis/basisnorm
        ev1 = ev1.view(1,512)
        cos = torch.mm(ev1.t(), basis.double())
        #cos = torch.mm(ev1.t(), basis.double())

        return cos

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
        #total = eigenvalues.sum()
        #eigsum = 0
        #index = 0
        #for i in range(1024):
        #    eigsum = eigsum + eigenvalues[1023-i]
        #    if eigsum >= total*k:
        #        index = 1023-i
        #        print(i)
        #        break;
        components = (eigenvectors[:, 1024-k:])
        return components

    def movingAVG(self, avg, x, count):
        if count == 1:
            avg = x.float()
            return avg
        avg = (count-1/count)*avg.float() + (1/count)*x.float()
        norm = torch.norm(avg)
        avg = avg/norm
        return avg


    def forward(self, anchor, positive, negative, targets, avg, idx ,size_average=True):
        count = idx+1
        data = torch.cat([anchor, positive, negative], dim=0)
        pca = self.PCA_scale(data, k=128)
        average = self.movingAVG(avg, pca, count)

        #lanchor = anchor.clone()
        #lpositive = positive.clone()
        #lnegative = negative.clone()

        #cos = self.findCos(pca.t()[0])
        #newpca=[]
        #for i in range(512):
        #    newpca.append(torch.mm(pca.t()[i].view(1,512), cos).view(512))
        #    #newpca.append((pca.t()*torch.cos(cos)).view(512, 512))

        anchor = torch.mm(anchor, average.float())
        positive = torch.mm(positive, average.float())
        negative = torch.mm(negative, average.float())

        #da = (anchor - lanchor).abs().sum(1)
        #dp = (positive - lpositive).abs().sum(1)
        #dn = (negative - lnegative).abs().sum(1)
        #Nlosses = (da + dp + dn)/512

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)
        gap = distance_positive - distance_negative
        Tlosses = F.relu(distance_positive - distance_negative + self.margin)
        return Tlosses.mean(), average

class SymmTripletPCALoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(SymmTripletPCALoss, self).__init__()
        self.margin = margin

    def findCos(self, ev1):
        basis = torch.zeros(512)
        basis[2] = 1
        basis.cuda()

        ev1norm = torch.norm(ev1)
        cos = torch.dot(ev1, basis) / ev1norm
        return cos

    def PCA_scale(self, x, k, center=True, scale=True):
        n, p = x.size()
        ones = torch.ones(n).view([n, 1]).cuda()
        h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
        H = torch.eye(n).cuda() - h
        X_center = torch.mm(H.double(), x.double())
        covariance = 1 / (n - 1) * torch.mm(X_center.t(), X_center).view(p, p)
        scaling = torch.sqrt(1 / torch.diag(covariance)).double() if scale else torch.ones(p).cuda().double()
        scaled_covariance = torch.mm(torch.diag(scaling).view(p, p), covariance)
        eigenvalues, eigenvectors = torch.eig(scaled_covariance, True)
        components = (eigenvectors[:, :k])
        return components

    def get_symmetric_point(self, anchor, pos):
        P = pos
        A = anchor  # basis
        A_norm = torch.norm(A)
        U = A / A_norm
        R = A + torch.sum(torch.mul((P - A), U), dim=1, keepdim=True) * U
        Q = 1 * (2 * R - P)
        Q = Q / torch.norm(Q) * A_norm
        return Q

    def forward(self, anchor, positive, negative, targets, size_average=True):
        data = torch.cat([anchor, positive, negative], dim=0)
        pca = self.PCA_scale(data, k=512)
        cos = self.findCos(pca[0])
        for i in range(512):
            pca[i] = torch.dot(pca[i], cos)

        anchor = torch.mm(anchor, pca.float())
        positive = torch.mm(positive, pca.float())
        negative = torch.mm(negative, pca.float())

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        sym1 = self.get_symmetric_point(anchor, positive)
        sym2 = self.get_symmetric_point(positive, anchor)

        pts_list = []
        pts_list.append(anchor)
        pts_list.append(positive)
        pts_list.append(sym1)
        pts_list.append(sym2)

        negative_distance_list = []

        for pts in pts_list:
            distance = (pts - negative).pow(2).sum(1)
            negative_distance_list.append(distance)

        list0 = negative_distance_list[0].reshape(1, negative_distance_list[0].shape[0]).transpose(0, 1)
        list1 = negative_distance_list[1].reshape(1, negative_distance_list[1].shape[0]).transpose(0, 1)
        list2 = negative_distance_list[2].reshape(1, negative_distance_list[2].shape[0]).transpose(0, 1)
        list3 = negative_distance_list[3].reshape(1, negative_distance_list[3].shape[0]).transpose(0, 1)
        dis_array = torch.cat((list0, list1, list2, list3), dim=1)
        min_array = torch.min(dis_array, dim=1)

        distance_negative = min_array.values

        gap = distance_positive - distance_negative
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
