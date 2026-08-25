from __future__ import annotations

import pytest

from tts_data_attribution.cli.errors import CommandError, require_flat_name


@pytest.mark.parametrize("value", ["", ".", "..", "../outside", "/tmp/outside"])
def test_flat_name_validation_rejects_unsafe_components(value: str) -> None:
    with pytest.raises(CommandError, match="single path component"):
        require_flat_name(value, "name")


def test_flat_name_validation_accepts_one_component() -> None:
    require_flat_name("run-0001", "name")
