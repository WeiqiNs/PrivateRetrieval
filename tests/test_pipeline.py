import json
from pathlib import Path

import pytest
from fixture import write_fixture

from privret import cli


def test_run_matches_plain_and_records_timings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "data"
    out = tmp_path / "out"
    write_fixture(data)
    argv = ["run", "--data", str(data), "--out", str(out), "--top-k", "5", "--max-queries", "2", "--threads", "3"]
    assert cli.main(argv) == 0

    plain = (out / "plain.run").read_bytes()
    assert (out / "encrypted.run").read_bytes() == plain
    lines = plain.decode().splitlines()
    assert lines[:5] == [
        "q0 Q0 d3 1 0 privret",
        "q0 Q0 d5 2 -127 privret",
        "q0 Q0 d1 3 -1270 privret",
        "q0 Q0 d2 4 -1270 privret",
        "q0 Q0 d4 5 -3556 privret",
    ]
    assert len(lines) == 10
    assert (out / "msk.bin").stat().st_mode & 0o777 == 0o600
    timings = [json.loads(line) for line in (out / "timing.jsonl").read_text().splitlines()]
    assert [timing["command"] for timing in timings] == ["setup", "encrypt", "keygen", "search", "plain"]
    assert (timings[3]["pairs"], timings[3]["threads"]) == (14, 3)
    assert "runs match" in capsys.readouterr().out
