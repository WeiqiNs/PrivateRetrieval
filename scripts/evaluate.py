import argparse
from pathlib import Path

import pytrec_eval

MEASURES = {"ndcg_cut.10", "recall.100"}


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for line in path.read_text().splitlines():
        query, _, doc, rel = line.split()
        qrels.setdefault(query, {})[doc] = int(rel)
    return qrels


def read_run(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    for line in path.read_text().splitlines():
        query, _, doc, _, score, _ = line.split()
        run.setdefault(query, {})[doc] = float(score)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Score TREC runs on their common queries with nDCG@10 and Recall@100.")
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()

    qrels = read_qrels(args.qrels)
    runs = {path: read_run(path) for path in args.runs}
    common = set(qrels).intersection(*runs.values())
    evaluator = pytrec_eval.RelevanceEvaluator({query: qrels[query] for query in common}, MEASURES)

    print("| run | queries | nDCG@10 | Recall@100 |")
    print("|---|---|---|---|")
    for path, run in runs.items():
        results = evaluator.evaluate({query: run[query] for query in common})
        ndcg = sum(result["ndcg_cut_10"] for result in results.values()) / len(results)
        recall = sum(result["recall_100"] for result in results.values()) / len(results)
        print(f"| {path.parent.name}/{path.name} | {len(results)} | {ndcg:.4f} | {recall:.4f} |")


if __name__ == "__main__":
    main()
