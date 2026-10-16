from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch.nn.functional import normalize

from privret.errors import PrivretError
from privret.evaluate import Qrels
from privret.export import Export, export_dim

MODEL = "Qwen/Qwen3-Embedding-0.6B"
DOC_PROMPT = ""
QUERY_PROMPT = "Instruct: Given a scientific claim, retrieve documents that support or refute the claim\nQuery:"
MAX_SEQ_LENGTH = 512


@dataclass(frozen=True)
class SciFact:
    doc_ids: tuple[str, ...]
    doc_texts: list[str]
    query_ids: tuple[str, ...]
    query_texts: list[str]
    qrels: Qrels


def load_scifact() -> SciFact:
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels: Qrels = {}
    for row in load_dataset("BeIR/scifact-qrels", split="test"):
        qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    kept = [row for row in queries if str(row["_id"]) in qrels]
    return SciFact(
        doc_ids=tuple(str(row["_id"]) for row in corpus),
        doc_texts=[f"{row['title']} {row['text']}" for row in corpus],
        query_ids=tuple(str(row["_id"]) for row in kept),
        query_texts=[row["text"] for row in kept],
        qrels=qrels,
    )


def encode(model: SentenceTransformer, *, texts: list[str], prompt: str, batch_size: int) -> torch.Tensor:
    return model.encode(texts, prompt=prompt, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)


def truncate(embeddings: torch.Tensor, dim: int) -> np.ndarray:
    return np.asarray(normalize(embeddings[:, :dim], p=2, dim=1).cpu().numpy(), dtype=np.float32)


def embed(*, dims: Sequence[int], batch_size: int, device: str | None) -> Export:
    data = load_scifact()
    model = SentenceTransformer(
        MODEL,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        model_kwargs={"dtype": torch.float32},
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    width = model.get_sentence_embedding_dimension()
    if max(dims) > width:
        raise PrivretError(f"{MODEL} embeds into {width} dimensions, fewer than {max(dims)}")
    docs = encode(model, texts=data.doc_texts, prompt=DOC_PROMPT, batch_size=batch_size)
    queries = encode(model, texts=data.query_texts, prompt=QUERY_PROMPT, batch_size=batch_size)
    return Export(
        qrels=data.qrels,
        dims=tuple(
            export_dim(
                doc_ids=data.doc_ids,
                query_ids=data.query_ids,
                docs=truncate(docs, dim),
                queries=truncate(queries, dim),
            )
            for dim in dims
        ),
    )
