import math
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pfe
from pfe.ipfe import opt

from privret.errors import DecryptError, FormatError
from privret.fileformat import Corpus, QueryKeys
from privret.ranking import Run, rank


def score_table(corpus: Corpus, keys: QueryKeys) -> pfe.DlogTable:
    bound = math.isqrt(corpus.max_norm_sq * keys.max_norm_sq)
    return pfe.DlogTable(opt.base(), -bound, bound)


def search_encrypted(corpus: Corpus, keys: QueryKeys, *, table: pfe.DlogTable, top_k: int, threads: int) -> Run:
    if corpus.dim != keys.dim:
        raise FormatError(f"documents have dim {corpus.dim} but queries have dim {keys.dim}")
    rows = []
    with ThreadPoolExecutor(threads) as pool:
        for query, key in zip(keys.ids, keys.records, strict=True):
            scores = list(pool.map(partial(opt.decrypt, table, opt.prepare(key)), corpus.records))
            if None in scores:
                raise DecryptError(
                    f"query {query} and doc {corpus.ids[scores.index(None)]} decrypt outside the score range; "
                    "were they made from the same master key?"
                )
            rows.append(scores)
    return rank(query_ids=keys.ids, doc_ids=corpus.ids, scores=np.array(rows, dtype=np.int64), top_k=top_k)
