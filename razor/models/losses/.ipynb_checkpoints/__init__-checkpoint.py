from .RDKD import RDKDLoss
from .cwd import ChannelWiseDivergence
from .dist_loss import DISTLoss
from .HRDKD import HRDKDLoss
from .ifvd import CriterionIFV
from .cirkd import CriterionKD, StudentSegContrast, CriterionMiniBatchCrossImagePair


__all__ = ['RDKDLoss','ChannelWiseDivergence','DISTLoss','HRDKDLoss','CriterionIFV','CriterionKD', 'StudentSegContrast', 'CriterionMiniBatchCrossImagePair']