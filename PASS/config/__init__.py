from enum import Enum, unique
import os
import os.path as osp
import sys

@unique
class KMMaskScheme(Enum):
    RandomSampling = 0
    Bernoulli = 1
    Topk = 2
    Learning = 3
    Origin = 4


@unique
class MaskMode(Enum):
    Anchor = 0
    Positive = 1
    Negative = 2


@unique
class RunningMode(Enum):
    BackboneTrain = 0
    BackboneTest = 1
    GatePreTrain = 2
    FineTuning = 3
    Test = 4


class MaskConfig:
    total_mac = 0
    skipped_mac = 0
    skipped_patch = 0
    total_patch = 0
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    mask = None
    mask_mode = MaskMode.Positive


m_cfg = MaskConfig()
