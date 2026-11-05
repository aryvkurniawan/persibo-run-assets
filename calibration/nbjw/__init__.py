"""Vendored HRNetV2-w48 backbone for the NBJW pitch-keypoint model.

Source: github.com/mguti97/No-Bells-Just-Whistles (model/cls_hrnet.py).
Only the architecture is vendored; trained weights live in weights/.
"""

from calibration.nbjw.cls_hrnet import get_cls_net

__all__ = ["get_cls_net"]
