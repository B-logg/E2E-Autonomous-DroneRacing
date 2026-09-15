"""GateNet: gate segmentation for real camera images (paper II-E, Appendix A).

Not used during simulation training -- the simulator renders masks directly.
GateNet exists for two jobs:

  1. at deployment, turn real RGB frames into the binary masks the policy eats;
  2. before that, produce the "real mask" domain that StochGAN is trained to
     imitate, which is what closes the visual sim-to-real gap.
"""

from .losses import GateNetLoss, iou, multiscale_loss
from .model import GateNet, build
from .preprocess import CameraCalibration, NominalRemapper, nominal_K

__all__ = [
    "GateNet",
    "GateNetLoss",
    "CameraCalibration",
    "NominalRemapper",
    "build",
    "iou",
    "multiscale_loss",
    "nominal_K",
]
