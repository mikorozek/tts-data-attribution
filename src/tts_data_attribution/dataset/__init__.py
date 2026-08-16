from .dailytalk import DailyTalkDataset, DailyTalkRecord
from .utterance import Utterance, UtteranceDataset

DATASETS = {"dailytalk": DailyTalkDataset}

__all__ = [
    "DATASETS",
    "DailyTalkDataset",
    "DailyTalkRecord",
    "Utterance",
    "UtteranceDataset",
]
