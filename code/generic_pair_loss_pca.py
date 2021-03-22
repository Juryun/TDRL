import torch

from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses.base_metric_loss_function import BaseMetricLossFunction


class GenericPairPCALoss(BaseMetricLossFunction):
    def __init__(self, mat_based_loss, **kwargs):
        super().__init__(**kwargs)
        self.loss_method = (
            self.mat_based_loss if mat_based_loss else self.pair_based_loss
        )

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
        index = 512-k
        #for i in range(512):
        #    eigsum = eigsum + eigenvalues[511-i]
        #    if eigsum >= total*k:
        #        index = 511-i
        #        break;
        components = (eigenvectors[:, index:])
        return components

    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = lmu.convert_to_pairs(indices_tuple, labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        PCA = self.PCA_scale(embeddings, k=128)
        embeddings = torch.mm(embeddings, PCA.float())
        mat = self.distance(embeddings)
        return self.loss_method(mat, labels, indices_tuple)

    def _compute_loss(self):
        raise NotImplementedError

    def mat_based_loss(self, mat, labels, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        return self._compute_loss(mat, pos_mask, neg_mask)

    def pair_based_loss(self, mat, labels, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]
        return self._compute_loss(pos_pair, neg_pair, indices_tuple)
