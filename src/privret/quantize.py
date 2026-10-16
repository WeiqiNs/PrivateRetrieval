from dataclasses import dataclass

import numpy as np

from privret.fileformat import INT8_LIMIT


@dataclass(frozen=True)
class Quantized:
    docs: np.ndarray
    queries: np.ndarray
    scale: float
    clipped: int


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
