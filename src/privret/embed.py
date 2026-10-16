import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch.nn.functional import normalize

from privret.fileformat import Vectors, encode_vectors
from privret.quantize import quantize
from privret.ranking import format_run, rank

MODEL = "Qwen/Qwen3-Embedding-0.6B"
DOC_PROMPT = ""
QUERY_PROMPT = "Instruct: Given a scientific claim, retrieve documents that support or refute the claim\nQuery:"
MAX_SEQ_LENGTH = 512
TOP_K = 100


@dataclass(frozen=True)
class SciFact:
    doc_ids: list[str]
    doc_texts: list[str]
    query_ids: list[str]
    query_texts: list[str]
    qrels: dict[str, dict[str, int]]


def load_scifact() -> SciFact:
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels: dict[str, dict[str, int]] = {}
    for row in load_dataset("BeIR/scifact-qrels", split="test"):
        qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    kept = [row for row in queries if str(row["_id"]) in qrels]
    return SciFact(
        doc_ids=[str(row["_id"]) for row in corpus],
        doc_texts=[f"{row['title']} {row['text']}" for row in corpus],
        query_ids=[str(row["_id"]) for row in kept],
        query_texts=[row["text"] for row in kept],
        qrels=qrels,
    )


def encode(model: SentenceTransformer, *, texts: list[str], prompt: str, batch_size: int) -> torch.Tensor:
    return model.encode(texts, prompt=prompt, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)


def truncate(embeddings: torch.Tensor, dim: int) -> np.ndarray:
    return np.asarray(normalize(embeddings[:, :dim], p=2, dim=1).cpu().numpy(), dtype=np.float32)


def embed(*, out: Path, dims: Sequence[int], batch_size: int, device: str | None) -> None:
    data = load_scifact()
    model = SentenceTransformer(
        MODEL,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        model_kwargs={"dtype": torch.float32},
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    docs = encode(model, texts=data.doc_texts, prompt=DOC_PROMPT, batch_size=batch_size)
    queries = encode(model, texts=data.query_texts, prompt=QUERY_PROMPT, batch_size=batch_size)

    out.mkdir(parents=True, exist_ok=True)
    with (out / "qrels.tsv").open("w") as qrels:
        for query, judged in data.qrels.items():
            for doc, rel in judged.items():
                qrels.write(f"{query} 0 {doc} {rel}\n")

    for dim in dims:
        folder = out / f"d{dim}"
        folder.mkdir(exist_ok=True)
        float_docs = truncate(docs, dim)
        float_queries = truncate(queries, dim)
        quantized = quantize(docs=float_docs, queries=float_queries)
        (folder / "docs.i8v").write_bytes(encode_vectors(Vectors(tuple(data.doc_ids), quantized.docs)))
        (folder / "queries.i8v").write_bytes(encode_vectors(Vectors(tuple(data.query_ids), quantized.queries)))
        run = rank(query_ids=data.query_ids, doc_ids=data.doc_ids, scores=float_queries @ float_docs.T, top_k=TOP_K)
        (folder / "float.run").write_bytes(format_run(run, tag="float", score_spec=".6f").encode())
        summary = {
            "dim": dim,
            "scale": quantized.scale,
            "max_abs_doc": float(np.abs(float_docs).max()),
            "clipped_query_entries": quantized.clipped,
            "docs": len(data.doc_ids),
            "queries": len(data.query_ids),
        }
        (folder / "quant.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))
