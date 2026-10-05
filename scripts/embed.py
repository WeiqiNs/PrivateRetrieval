import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

MODEL = "Qwen/Qwen3-Embedding-0.6B"
DOC_PROMPT = ""
QUERY_PROMPT = "Instruct: Given a scientific claim, retrieve documents that support or refute the claim\nQuery:"
MAX_SEQ_LENGTH = 512
INT8_LIMIT = 127
TOP_K = 100


@dataclass(frozen=True)
class SciFact:
    doc_ids: list[str]
    doc_texts: list[str]
    query_ids: list[str]
    query_texts: list[str]
    qrels: dict[str, dict[str, int]]


@dataclass(frozen=True)
class Quantized:
    docs: np.ndarray
    queries: np.ndarray
    scale: float
    clipped: int


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


def embed(model: SentenceTransformer, *, texts: list[str], prompt: str, batch_size: int) -> torch.Tensor:
    return model.encode(texts, prompt=prompt, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)


def truncate(embeddings: torch.Tensor, dim: int) -> np.ndarray:
    return F.normalize(embeddings[:, :dim], p=2, dim=1).float().cpu().numpy()


def quantize(*, docs: np.ndarray, queries: np.ndarray) -> Quantized:
    scale = INT8_LIMIT / float(np.abs(docs).max())
    scaled_docs = np.rint(docs.astype(np.float64) * scale)
    scaled_queries = np.rint(queries.astype(np.float64) * scale)
    return Quantized(
        docs=scaled_docs.astype(np.int8),
        queries=np.clip(scaled_queries, -INT8_LIMIT, INT8_LIMIT).astype(np.int8),
        scale=scale,
        clipped=int((np.abs(scaled_queries) > INT8_LIMIT).sum()),
    )


def write_vectors(path: Path, *, ids: list[str], values: np.ndarray) -> None:
    count, dim = values.shape
    assert count == len(ids)
    with path.open("wb") as out:
        out.write(struct.pack("<4sIII", b"PRVC", 1, dim, count))
        for id_ in ids:
            encoded = id_.encode("utf-8")
            out.write(struct.pack("<H", len(encoded)) + encoded)
        out.write(values.astype(np.int8).tobytes())


def write_run(path: Path, *, query_ids: list[str], doc_ids: list[str], scores: np.ndarray, tag: str) -> None:
    id_rank = np.argsort(np.argsort(np.array([doc.encode("utf-8") for doc in doc_ids], dtype=object)))
    with path.open("w") as out:
        for query, row in zip(query_ids, scores):
            order = np.lexsort((id_rank, -row))[:TOP_K]
            for rank, doc in enumerate(order, start=1):
                out.write(f"{query} Q0 {doc_ids[doc]} {rank} {row[doc]:.6f} {tag}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed SciFact with Qwen3-Embedding-0.6B and export int8 vectors.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = load_scifact()
    model = SentenceTransformer(MODEL, device=args.device, model_kwargs={"dtype": torch.float32})
    model.max_seq_length = MAX_SEQ_LENGTH
    docs = embed(model, texts=data.doc_texts, prompt=DOC_PROMPT, batch_size=args.batch_size)
    queries = embed(model, texts=data.query_texts, prompt=QUERY_PROMPT, batch_size=args.batch_size)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "qrels.tsv").open("w") as out:
        for query, judged in data.qrels.items():
            for doc, rel in judged.items():
                out.write(f"{query} 0 {doc} {rel}\n")

    for dim in args.dims:
        folder = args.out / f"d{dim}"
        folder.mkdir(exist_ok=True)
        float_docs = truncate(docs, dim)
        float_queries = truncate(queries, dim)
        quantized = quantize(docs=float_docs, queries=float_queries)
        write_vectors(folder / "docs.i8v", ids=data.doc_ids, values=quantized.docs)
        write_vectors(folder / "queries.i8v", ids=data.query_ids, values=quantized.queries)
        write_run(
            folder / "float.run",
            query_ids=data.query_ids,
            doc_ids=data.doc_ids,
            scores=float_queries @ float_docs.T,
            tag="float",
        )
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


if __name__ == "__main__":
    main()
