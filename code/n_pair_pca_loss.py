import torch

from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses.base_metric_loss_function import BaseMetricLossFunction


class NPairsPCALoss(BaseMetricLossFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_to_recordable_attributes(name="num_pairs", is_stat=True)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

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

    def compute_loss(self, embeddings, labels, indices_tuple):
        anchor_idx, positive_idx = lmu.convert_to_pos_pairs_with_unique_labels(
            indices_tuple, labels
        )
        self.num_pairs = len(anchor_idx)
        if self.num_pairs == 0:
            return self.zero_losses()
        data = torch.cat([embeddings[anchor_idx], embeddings[positive_idx]], dim=0)
        PCA = self.PCA_scale(data, k=128)
        new_anchor = torch.mm(embeddings[anchor_idx], PCA.float())
        new_positive = torch.mm(embeddings[positive_idx], PCA.float())
        anchors, positives = new_anchor, new_positive
        targets = torch.arange(self.num_pairs).to(embeddings.device)
        sim_mat = self.distance(anchors, positives)
        if not self.distance.is_inverted:
            sim_mat = -sim_mat
        return {
            "loss": {
                "losses": self.cross_entropy(sim_mat, targets),
                "indices": anchor_idx,
                "reduction_type": "element",
            }
        }

    def get_default_distance(self):
        return DotProductSimilarity()
