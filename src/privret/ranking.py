from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from privret.errors import FormatError
from privret.fileformat import Vectors

type Run = tuple[Ranking, ...]


@dataclass(frozen=True)
class Hit:
    doc: str
    score: int | float


@dataclass(frozen=True)
class Ranking:
    query: str
    hits: tuple[Hit, ...]


def rank(*, query_ids: Sequence[str], doc_ids: Sequence[str], scores: np.ndarray, top_k: int) -> Run:
    id_rank = np.argsort(np.argsort(np.array(doc_ids, dtype=object)))
    return tuple(
        Ranking(query, tuple(Hit(doc_ids[doc], row[doc].item()) for doc in np.lexsort((id_rank, -row))[:top_k]))
        for query, row in zip(query_ids, scores, strict=True)
    )


def search_plain(docs: Vectors, queries: Vectors, *, top_k: int) -> Run:
    if docs.dim != queries.dim:
        raise FormatError(f"documents have dim {docs.dim} but queries have dim {queries.dim}")
    scores = queries.values.astype(np.int64) @ docs.values.astype(np.int64).T
    return rank(query_ids=queries.ids, doc_ids=docs.ids, scores=scores, top_k=top_k)


def format_run(run: Run, *, tag: str = "privret", score_spec: str = "d") -> str:
    return "".join(
        f"{ranking.query} Q0 {hit.doc} {position} {hit.score:{score_spec}} {tag}\n"
        for ranking in run
        for position, hit in enumerate(ranking.hits, start=1)
    )
