import numpy as np
import pytest
from pfe.ipfe import opt

from privret.client import seal
from privret.errors import DecryptError, FormatError
from privret.fileformat import CIPHERTEXTS, KEYS, Vectors, decode_sealed, encode_sealed
from privret.ranking import Hit, search_plain
from privret.server import score_table, search_encrypted

DIM = 6


def sample() -> tuple[Vectors, Vectors]:
    rng = np.random.default_rng(7)
    random_docs = rng.integers(-127, 128, size=(8, DIM))
    twin = random_docs[-1].copy()
    random_docs[0] = 127
    docs = Vectors(
        ("d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d10"),
        np.array([*random_docs[:2], twin, *random_docs[2:], twin], dtype=np.int8),
    )
    queries = Vectors(
        ("q0", "q1", "q2", "q3", "q4"),
        np.array([np.full(DIM, -127), twin, *rng.integers(-127, 128, size=(3, DIM))], dtype=np.int8),
    )
    return docs, queries


DOCS, QUERIES = sample()


@pytest.mark.parametrize("threads", [1, 3, 16])
def test_encrypted_search_matches_plain_search_exactly(threads: int) -> None:
    msk = opt.setup(DIM)
    corpus = decode_sealed(encode_sealed(seal(msk, DOCS, CIPHERTEXTS)), CIPHERTEXTS)
    keys = decode_sealed(encode_sealed(seal(msk, QUERIES, KEYS)), KEYS)
    table = score_table(corpus, keys)

    plain = search_plain(DOCS, QUERIES, top_k=10)
    assert plain[0].hits[-1] == Hit("d0", -96774)
    ranked = [hit.doc for hit in plain[1].hits]
    assert ranked[ranked.index("d10") + 1] == "d2"
    assert search_encrypted(corpus, keys, table=table, top_k=10, threads=threads) == plain


def test_rejects_keys_from_another_master_key_or_dim() -> None:
    corpus = seal(opt.setup(DIM), DOCS, CIPHERTEXTS)
    foreign_keys = seal(opt.setup(DIM), QUERIES, KEYS)
    with pytest.raises(DecryptError, match="query q0 and doc d0 decrypt outside the score range"):
        search_encrypted(corpus, foreign_keys, table=score_table(corpus, foreign_keys), top_k=3, threads=2)
    short = Vectors(("s0",), np.ones((1, DIM - 1), dtype=np.int8))
    short_keys = seal(opt.setup(DIM - 1), short, KEYS)
    with pytest.raises(FormatError, match="documents have dim 6 but queries have dim 5"):
        search_encrypted(corpus, short_keys, table=score_table(corpus, short_keys), top_k=3, threads=2)
