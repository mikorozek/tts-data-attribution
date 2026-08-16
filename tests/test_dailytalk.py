from __future__ import annotations

from pathlib import Path

from conftest import write_dailytalk_fixture

from tts_data_attribution.dataset import DailyTalkDataset


def test_dailytalk_loads_utterances_from_the_official_layout(tmp_path: Path) -> None:
    write_dailytalk_fixture(tmp_path)

    dataset = DailyTalkDataset(tmp_path)

    assert dataset.data_root == tmp_path
    assert [record.id for record in dataset.get_records()] == ["2-0", "2-1", "10-0"]
    first = dataset.get_records_by_ids(["2-0"])[0]
    assert first.text == "First"
    assert first.speaker == "0"
    assert first.dialogue == "2"
    assert first.audio_path == "data/2/0_0_d2.wav"
    assert (
        dataset.get_records_by_ids(["10-0"])[0].text
        == "Transcript from the utterance file"
    )


def test_dailytalk_gets_selected_records_by_their_stable_ids(tmp_path: Path) -> None:
    write_dailytalk_fixture(tmp_path)

    records = DailyTalkDataset(tmp_path).get_records_by_ids(["10-0", "2-0"])

    assert [record.id for record in records] == ["10-0", "2-0"]
