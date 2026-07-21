import torch

from ..reducers import AvgNonZeroReducer
from ..utils import loss_and_miner_utils as lmu
from ..losses.base_metric_loss_function import BaseMetricLossFunction


class TripletMarginMMDLoss(BaseMetricLossFunction):
    """
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
    """

    def __init__(
        self,
        margin=0.05,
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="all",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)

    def compute_loss(self, embeddings, labels, indices_tuple, embs_list, lambda1=0.3):
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, t_per_anchor=self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        mat = self.distance(embeddings)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        if self.swap:
            pn_dists = mat[positive_idx, negative_idx]
            an_dists = self.distance.smallest_dist(an_dists, pn_dists)

        current_margins = self.distance.margin(an_dists, ap_dists)
        if self.smooth_loss:
            loss = torch.log(1 + torch.exp(-current_margins))
        else:
            loss = torch.nn.functional.relu(-current_margins + self.margin)
        loss = loss + lambda1*self.compute_MMD_loss(embs_list)
        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }
        
    def compute_MMD_loss(self, embs):
        embs_msum= [torch.sum(e, dim=0)/e.shape[0] for e in embs]
        n = 0
        MMD_loss = 0
        for i in range(len(embs_msum)):
            for j in range(i+1, len(embs_msum)):
                MMD_loss = MMD_loss + torch.square(torch.norm(embs_msum[i]-embs_msum[j]))
                n +=1
        MMD_loss = MMD_loss/n
        return MMD_loss
        
    def get_default_reducer(self):
        return AvgNonZeroReducer()
