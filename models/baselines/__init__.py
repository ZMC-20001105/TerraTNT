"""
基线模型集合
"""
from .base_predictor import BasePredictor
from .ynet import YNet
from .pecnet import PECNet
from .trajectron import TrajectronPP
from .social_lstm import SocialLSTM
from .constant_velocity import ConstantVelocity

__all__ = [
    'BasePredictor',
    'YNet',
    'PECNet',
    'TrajectronPP',
    'SocialLSTM',
    'ConstantVelocity'
]
