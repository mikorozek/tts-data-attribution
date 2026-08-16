from __future__ import annotations

from pathlib import Path

from conftest import write_dailytalk_fixture

from tts_data_attribution.dataset import DailyTalkDataset, Utterance, UtteranceDataset


def test_dailytalk_loads_utterances_from_the_official_layout(tmp_path: Path) -> None:
    write_dailytalk_fixture(tmp_path)

    dataset = DailyTalkDataset(tmp_path)

    assert isinstance(dataset, UtteranceDataset)
    assert dataset.data_root == tmp_path
    assert [utterance.id for utterance in dataset] == ["2-0", "2-1", "10-0"]
    assert dataset[0] == Utterance(
        id="2-0",
        text="First",
        speaker="0",
        dialogue="2",
        audio_path="data/2/0_0_d2.wav",
    )
    assert dataset[2].text == "Transcript from the utterance file"
    assert dataset[2].audio_codes is None
