from enum import Enum, unique


@unique
class KMMaskScheme(Enum):
    RandomSampling = 0
    Bernoulli = 1
    Topk = 2
    Learning = 3
    Origin = 4


@unique
class MaskMode(Enum):
    Origin = 0
    Positive = 1
    Negative = 2


@unique
class RunningMode(Enum):
    BackboneTrain = 0
    BackboneTest = 1
    GatePreTrain = 2
    FineTuning = 3
    Test = 4


class Config:
    total_mac = 0
    skipped_mac = 0
    skipped_patch = 0
    total_patch = 0


cfg = Config()
