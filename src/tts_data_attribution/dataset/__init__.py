from .dailytalk import DailyTalkDataset
from .utterance import Utterance, UtteranceDataset

DATASETS = {"dailytalk": DailyTalkDataset}

__all__ = ["DATASETS", "DailyTalkDataset", "Utterance", "UtteranceDataset"]
