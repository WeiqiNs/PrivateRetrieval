from dataclasses import dataclass

import numpy as np

from privret.evaluate import Qrels
from privret.fileformat import Vectors
from privret.quantize import quantize
from privret.ranking import Run, rank

TOP_K = 100


@dataclass(frozen=True)
class DimExport:
    docs: Vectors
    queries: Vectors
    float_run: Run
    scale: float
    max_abs_doc: float
    clipped: int


@dataclass(frozen=True)
class Export:
    qrels: Qrels
    dims: tuple[DimExport, ...]


def export_dim(
    *, doc_ids: tuple[str, ...], query_ids: tuple[str, ...], docs: np.ndarray, queries: np.ndarray
) -> DimExport:
    quantized = quantize(docs=docs, queries=queries)
    return DimExport(
        docs=Vectors(doc_ids, quantized.docs),
        queries=Vectors(query_ids, quantized.queries),
        float_run=rank(query_ids=query_ids, doc_ids=doc_ids, scores=queries @ docs.T, top_k=TOP_K),
        scale=quantized.scale,
        max_abs_doc=quantized.max_abs_doc,
        clipped=quantized.clipped,
    )
