from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytrec_eval

from privret.errors import FormatError, PrivretError

type Qrels = dict[str, dict[str, int]]

MEASURES = {"ndcg_cut.10", "recall.100"}


@dataclass(frozen=True)
class RunQuality:
    name: str
    queries: int
    ndcg_at_10: float
    recall_at_100: float


def fields(path: Path, count: int) -> list[list[str]]:
    rows = [line.split() for line in path.read_text().splitlines()]
    for number, row in enumerate(rows, start=1):
        if len(row) != count:
            raise FormatError(f"{path}:{number} has {len(row)} fields, not {count}")
    return rows


def read_qrels(path: Path) -> Qrels:
    qrels: Qrels = {}
    for query, _, doc, rel in fields(path, 4):
        try:
            qrels.setdefault(query, {})[doc] = int(rel)
        except ValueError as err:
            raise FormatError(f"{path} has a non-integer relevance {rel!r}") from err
    return qrels


def format_qrels(qrels: Qrels) -> str:
    return "".join(f"{query} 0 {doc} {rel}\n" for query, judged in qrels.items() for doc, rel in judged.items())


def read_run(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    for query, _, doc, _, score, _ in fields(path, 6):
        try:
            run.setdefault(query, {})[doc] = float(score)
        except ValueError as err:
            raise FormatError(f"{path} has a non-numeric score {score!r}") from err
    return run


def evaluate(
    qrels: Mapping[str, dict[str, int]], runs: Mapping[str, Mapping[str, dict[str, float]]]
) -> list[RunQuality]:
    common = set(qrels).intersection(*runs.values())
    if not common:
        raise PrivretError("the qrels and the runs share no query")
    evaluator = pytrec_eval.RelevanceEvaluator({query: qrels[query] for query in common}, MEASURES)
    qualities = []
    for name, run in runs.items():
        results = evaluator.evaluate({query: run[query] for query in common})
        qualities.append(
            RunQuality(
                name=name,
                queries=len(results),
                ndcg_at_10=sum(result["ndcg_cut_10"] for result in results.values()) / len(results),
                recall_at_100=sum(result["recall_100"] for result in results.values()) / len(results),
            )
        )
    return qualities


def format_table(qualities: list[RunQuality]) -> str:
    rows = [
        f"| {quality.name} | {quality.queries} | {quality.ndcg_at_10:.4f} | {quality.recall_at_100:.4f} |"
        for quality in qualities
    ]
    return "\n".join(["| run | queries | nDCG@10 | Recall@100 |", "|---|---|---|---|", *rows]) + "\n"
