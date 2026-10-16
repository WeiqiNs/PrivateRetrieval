import numpy as np

from privret.ranking import Hit, Ranking, format_run, rank


def test_orders_by_score_then_doc_id() -> None:
    scores = np.array([[5, 5, 9, -1]], dtype=np.int64)
    top = Hit("c", 9), Hit("a", 5), Hit("b", 5)
    assert rank(query_ids=["q"], doc_ids=["b", "a", "c", "d"], scores=scores, top_k=3) == (Ranking("q", top),)
    assert rank(query_ids=["q"], doc_ids=["b", "a", "c", "d"], scores=scores, top_k=10) == (
        Ranking("q", (*top, Hit("d", -1))),
    )


def test_formats_trec_lines() -> None:
    run = Ranking("q1", (Hit("d3", 12), Hit("d1", -4))), Ranking("q2", (Hit("d1", 0),))
    assert format_run(run) == "q1 Q0 d3 1 12 privret\nq1 Q0 d1 2 -4 privret\nq2 Q0 d1 1 0 privret\n"
    float_run = (Ranking("q1", (Hit("d1", 0.5),)),)
    assert format_run(float_run, tag="float", score_spec=".6f") == "q1 Q0 d1 1 0.500000 float\n"
