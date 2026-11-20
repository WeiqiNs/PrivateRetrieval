from pathlib import Path

import pytest

from privret.errors import FormatError, PrivretError
from privret.evaluate import evaluate, read_qrels, read_run


def test_evaluate_rejects_runs_sharing_no_query_with_the_qrels() -> None:
    with pytest.raises(PrivretError, match="share no query"):
        evaluate({"q1": {"d1": 1}}, {"a": {"q2": {"d1": 1.0}}})


def test_reading_rejects_malformed_lines(tmp_path: Path) -> None:
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\n")
    with pytest.raises(FormatError, match="has 3 fields, not 4"):
        read_qrels(qrels)
    qrels.write_text("q1 0 d1 1 extra\n")
    with pytest.raises(FormatError, match="has 5 fields, not 4"):
        read_qrels(qrels)
    qrels.write_text("q1 0 d1 high\n")
    with pytest.raises(FormatError, match="non-integer relevance"):
        read_qrels(qrels)
    run = tmp_path / "a.run"
    run.write_text("q1 Q0 d1 1 high privret\n")
    with pytest.raises(FormatError, match="non-numeric score"):
        read_run(run)
