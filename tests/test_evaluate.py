from pathlib import Path

import pytest

from privret.errors import FormatError, PrivretError
from privret.evaluate import evaluate, read_qrels, read_run


def test_evaluate_scores_runs_on_shared_queries() -> None:
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    runs = {
        "a": {"q1": {"d1": 2.0, "d9": 1.0}, "q2": {"d9": 1.0}},
        "b": {"q1": {"d9": 1.0, "d1": 0.5}},
    }
    a, b = evaluate(qrels, runs)
    assert (a.name, a.queries, a.ndcg_at_10, a.recall_at_100) == ("a", 1, 1.0, 1.0)
    assert (b.name, b.queries, b.recall_at_100) == ("b", 1, 1.0)
    assert b.ndcg_at_10 == pytest.approx(0.6309, abs=1e-4)


def test_evaluate_rejects_runs_sharing_no_query_with_the_qrels() -> None:
    with pytest.raises(PrivretError, match="share no query"):
        evaluate({"q1": {"d1": 1}}, {"a": {"q2": {"d1": 1.0}}})


def test_reading_rejects_malformed_lines(tmp_path: Path) -> None:
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\n")
    with pytest.raises(FormatError, match="has 3 fields, not 4"):
        read_qrels(qrels)
    qrels.write_text("q1 0 d1 high\n")
    with pytest.raises(FormatError, match="non-integer relevance"):
        read_qrels(qrels)
    run = tmp_path / "a.run"
    run.write_text("q1 Q0 d1 1 high privret\n")
    with pytest.raises(FormatError, match="non-numeric score"):
        read_run(run)
