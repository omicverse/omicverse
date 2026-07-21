from ..losses.angular_loss import AngularLoss
from ..losses.arcface_loss import ArcFaceLoss
from ..losses.base_metric_loss_function import BaseMetricLossFunction, MultipleLosses
from ..losses.circle_loss import CircleLoss
from ..losses.contrastive_loss import ContrastiveLoss
from ..losses.cosface_loss import CosFaceLoss
from ..losses.cross_batch_memory import CrossBatchMemory
from ..losses.fast_ap_loss import FastAPLoss
from ..losses.generic_pair_loss import GenericPairLoss
from ..losses.intra_pair_variance_loss import IntraPairVarianceLoss
from ..losses.large_margin_softmax_loss import LargeMarginSoftmaxLoss
from ..losses.lifted_structure_loss import GeneralizedLiftedStructureLoss, LiftedStructureLoss
from ..losses.margin_loss import MarginLoss
from ..losses.mixins import EmbeddingRegularizerMixin, WeightRegularizerMixin
from ..losses.multi_similarity_loss import MultiSimilarityLoss
from ..losses.n_pairs_loss import NPairsLoss
from ..losses.nca_loss import NCALoss
from ..losses.normalized_softmax_loss import NormalizedSoftmaxLoss
from ..losses.ntxent_loss import NTXentLoss
from ..losses.proxy_anchor_loss import ProxyAnchorLoss
from ..losses.proxy_losses import ProxyNCALoss
from ..losses.signal_to_noise_ratio_losses import SignalToNoiseRatioContrastiveLoss
from ..losses.soft_triple_loss import SoftTripleLoss
from ..losses.sphereface_loss import SphereFaceLoss
from ..losses.triplet_margin_loss import TripletMarginLoss
from ..losses.triplet_margin_mmd_loss import TripletMarginMMDLoss
from ..losses.tuplet_margin_loss import TupletMarginLoss
