import json
from pathlib import Path

import numpy as np
import pytest
from fixture import write_fixture

from privret import cli, pipeline
from privret.evaluate import read_qrels
from privret.export import Export, export_dim
from privret.fileformat import decode_vectors


def test_run_matches_plain_and_records_timings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "data"
    out = tmp_path / "out"
    write_fixture(data)
    out.mkdir()
    (out / "msk.bin").touch(mode=0o644)
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


def test_embed_writes_vectors_float_run_summary_and_qrels(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    qrels = {"q1": {"d0": 1, "d2": 2}, "q0": {"d1": 1}}
    dim_export = export_dim(
        doc_ids=("d0", "d1", "d2"),
        query_ids=("q0", "q1"),
        docs=np.array([[0.5, -0.25], [0.1, 0.0], [0.0, 0.5]], dtype=np.float32),
        queries=np.array([[0.6, -0.1], [0.5, 0.25]], dtype=np.float32),
    )
    pipeline.embed(export=Export(qrels=qrels, dims=(dim_export,)), out=tmp_path)

    assert (tmp_path / "qrels.tsv").read_text() == "q1 0 d0 1\nq1 0 d2 2\nq0 0 d1 1\n"
    assert read_qrels(tmp_path / "qrels.tsv") == qrels
    folder = tmp_path / "d2"
    docs = decode_vectors((folder / "docs.i8v").read_bytes())
    assert docs.ids == ("d0", "d1", "d2")
    assert docs.values.tolist() == [[127, -64], [25, 0], [0, 127]]
    queries = decode_vectors((folder / "queries.i8v").read_bytes())
    assert queries.ids == ("q0", "q1")
    assert queries.values.tolist() == [[127, -25], [127, 64]]
    assert (folder / "float.run").read_text().splitlines() == [
        "q0 Q0 d0 1 0.325000 float",
        "q0 Q0 d1 2 0.060000 float",
        "q0 Q0 d2 3 -0.050000 float",
        "q1 Q0 d0 1 0.187500 float",
        "q1 Q0 d2 2 0.125000 float",
        "q1 Q0 d1 3 0.050000 float",
    ]
    expected = {"dim": 2, "scale": 254.0, "max_abs_doc": 0.5, "clipped_query_entries": 1, "docs": 3, "queries": 2}
    assert (folder / "quant.json").read_text() == json.dumps(expected, indent=2) + "\n"
    assert capsys.readouterr().out == json.dumps(expected) + "\n"


def test_evaluate_prints_one_row_per_run_and_rejects_shared_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("q1 0 d1 1\nq2 0 d2 1\n")
    exact = tmp_path / "exact" / "x.run"
    partial = tmp_path / "partial" / "x.run"
    exact.parent.mkdir()
    partial.parent.mkdir()
    exact.write_text("q1 Q0 d1 1 2 privret\nq2 Q0 d2 1 1 privret\n")
    partial.write_text("q1 Q0 d9 1 1.0 float\nq1 Q0 d1 2 0.5 float\n")
    assert cli.main(["evaluate", "--qrels", str(qrels), str(exact), str(partial)]) == 0
    assert capsys.readouterr().out == (
        "| run | queries | nDCG@10 | Recall@100 |\n"
        "|---|---|---|---|\n"
        "| exact/x.run | 1 | 1.0000 | 1.0000 |\n"
        "| partial/x.run | 1 | 0.6309 | 1.0000 |\n"
    )
    shadow = tmp_path / "other" / "exact" / "x.run"
    shadow.parent.mkdir(parents=True)
    shadow.write_text(partial.read_text())
    assert cli.main(["evaluate", "--qrels", str(qrels), str(exact), str(shadow)]) == 1
    assert capsys.readouterr().err == "privret: two runs are named exact/x.run\n"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["run", "--data", "missing", "--out", "out", "--top-k", "1"], "No such file or directory"),
        (
            ["search", "--corpus", "data/queries.i8v", "--keys", "data/queries.i8v", "--top-k", "1", "--out", "o"],
            "expected a PRCT file",
        ),
    ],
    ids=["os error", "format error"],
)
def test_main_reports_errors_and_exits_1(
    argv: list[str], message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_fixture(tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert err.startswith("privret: ")
    assert message in err


@pytest.mark.parametrize(
    "argv",
    [["--top-k", "0"], ["--top-k", "x"], ["--threads", "0"], ["--dim", "4097"]],
    ids=["top-k 0", "top-k not a number", "threads 0", "dim 4097"],
)
def test_parser_rejects_out_of_range_numbers(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    command = "setup" if argv[0] == "--dim" else "search"
    with pytest.raises(SystemExit) as info:
        cli.parser().parse_args([command, *argv])
    assert info.value.code == 2
    assert f"argument {argv[0]}" in capsys.readouterr().err
