import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses
from triplet_pca_margin_loss import TripletPCAMarginLoss
from contrastive_pca_loss import ContrastivePCALoss
from n_pair_pca_loss import NPairsPCALoss

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss


class Proxy_PCA_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha


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
        for i in range(512):
            eigsum = eigsum + eigenvalues[511-i]
            if eigsum >= total*k:
                index = 511-i
                nm = i
                break;
        components = (eigenvectors[:, 512-k:])
        return components, k


    def forward(self, X, T):
        pca, nm = self.PCA_scale(X, k=256)
        X = torch.mm(X, pca.float())
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss

# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


class MultiSimilarityPCALoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityPCALoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50

        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)

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
       #eigsum = 0
        index = 0
        #for i in range(512):
        #    eigsum = eigsum + eigenvalues[511-i]
        #    if eigsum >= total*k:
        #        index = 511-i
        #        break;
        components = (eigenvectors[:, 512-k:])
        return components

    def forward(self, embeddings, labels):
        PCA = self.PCA_scale(embeddings, k=128)
        embeddings = torch.mm(embeddings, PCA.float())
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class Contrastive_PCALoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(Contrastive_PCALoss, self).__init__()
        self.margin = margin
        self.loss_func = ContrastivePCALoss(neg_margin=self.margin)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss

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
        # Proxy Anchor Initialization
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets='semihard')
        self.loss_func = TripletPCAMarginLoss(margin=self.margin)

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss

class TripletPCA2Loss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletPCA2Loss, self).__init__()
        self.margin = margin
        self.k = 0.75
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets='semihard')
        self.loss_func = losses.TripletMarginLoss(margin=self.margin)

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
        index = 511-k
        for i in range(512):
            eigsum = eigsum + eigenvalues[511-i]
            if eigsum >= total*k:
                index = 511-i
                break;
        components = (eigenvectors[:, index:])
        return components

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        PCA = self.PCA_scale(embeddings, k=0.6)
        embeddings = torch.mm(embeddings, PCA.float())
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss

class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(embedding_reg_weight=0)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class NPairPCALoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairPCALoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = NPairsPCALoss()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss