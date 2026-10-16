import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from pfe.ipfe import opt

from privret import client, server
from privret.errors import RunMismatchError
from privret.evaluate import evaluate, format_table, read_qrels, read_run
from privret.fileformat import (
    CIPHERTEXTS,
    KEYS,
    Record,
    SealedKind,
    Vectors,
    decode_master_key,
    decode_sealed,
    decode_vectors,
    encode_master_key,
    encode_sealed,
)
from privret.ranking import format_run, search_plain


@dataclass(frozen=True)
class Timings:
    stream: TextIO

    def record(self, command: str, **fields: float) -> None:
        self.stream.write(json.dumps({"command": command, **fields}) + "\n")
        self.stream.flush()


@dataclass(frozen=True)
class Sealing[R: Record]:
    kind: SealedKind[R]
    command: str
    items: str
    ms_per_item: str


ENCRYPT = Sealing(CIPHERTEXTS, "encrypt", "docs", "ms_per_doc")
KEYGEN = Sealing(KEYS, "keygen", "queries", "ms_per_query")


def read_vectors(path: Path, limit: int | None) -> Vectors:
    vectors = decode_vectors(path.read_bytes())
    return vectors if limit is None else vectors.first(limit)


def write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as out:
        os.fchmod(out.fileno(), 0o600)
        out.write(data)


def setup(*, dim: int, msk: Path, timings: Timings) -> None:
    start = time.perf_counter()
    key = opt.setup(dim)
    seconds = time.perf_counter() - start
    write_private(msk, encode_master_key(key))
    timings.record("setup", dim=dim, seconds=seconds)


def seal_file[R: Record](
    *, sealing: Sealing[R], msk: Path, vectors: Path, out: Path, limit: int | None, timings: Timings
) -> None:
    key = decode_master_key(msk.read_bytes())
    plain = read_vectors(vectors, limit)
    start = time.perf_counter()
    sealed = client.seal(key, plain, sealing.kind)
    seconds = time.perf_counter() - start
    data = encode_sealed(sealed)
    write(out, data)
    count = len(sealed.ids)
    timings.record(
        sealing.command,
        dim=sealed.dim,
        **{sealing.items: count},
        seconds=seconds,
        **{sealing.ms_per_item: 1000 * seconds / count},
        bytes=len(data),
        record_bytes=len(bytes(sealed.records[0])),
    )


def search(*, corpus: Path, keys: Path, out: Path, top_k: int, threads: int, timings: Timings) -> None:
    start = time.perf_counter()
    docs = decode_sealed(corpus.read_bytes(), CIPHERTEXTS, threads=threads)
    queries = decode_sealed(keys.read_bytes(), KEYS, threads=threads)
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    table = server.score_table(docs, queries)
    table_ms = 1000 * (time.perf_counter() - start)
    start = time.perf_counter()
    run = server.search_encrypted(docs, queries, table=table, top_k=top_k, threads=threads)
    seconds = time.perf_counter() - start
    write(out, format_run(run).encode())
    pairs = len(docs.ids) * len(queries.ids)
    timings.record(
        "search",
        dim=docs.dim,
        docs=len(docs.ids),
        queries=len(queries.ids),
        pairs=pairs,
        threads=threads,
        load_seconds=load_seconds,
        table_ms=table_ms,
        seconds=seconds,
        ms_per_pair=1000 * seconds / pairs,
    )


def plain(*, docs: Path, queries: Path, out: Path, top_k: int, max_queries: int | None, timings: Timings) -> None:
    doc_vectors = decode_vectors(docs.read_bytes())
    query_vectors = read_vectors(queries, max_queries)
    start = time.perf_counter()
    run = search_plain(doc_vectors, query_vectors, top_k=top_k)
    seconds = time.perf_counter() - start
    write(out, format_run(run).encode())
    timings.record("plain", docs=len(doc_vectors.ids), queries=len(query_vectors.ids), seconds=seconds)


def run(*, data: Path, out: Path, top_k: int, max_queries: int | None, threads: int) -> None:
    docs = data / "docs.i8v"
    queries = data / "queries.i8v"
    msk = out / "msk.bin"
    corpus = out / "corpus.ct"
    keys = out / "queries.sk"
    encrypted_run = out / "encrypted.run"
    plain_run = out / "plain.run"
    dim = decode_vectors(docs.read_bytes()).dim
    out.mkdir(parents=True, exist_ok=True)
    with (out / "timing.jsonl").open("w") as stream:
        timings = Timings(stream)
        setup(dim=dim, msk=msk, timings=timings)
        seal_file(sealing=ENCRYPT, msk=msk, vectors=docs, out=corpus, limit=None, timings=timings)
        seal_file(sealing=KEYGEN, msk=msk, vectors=queries, out=keys, limit=max_queries, timings=timings)
        search(corpus=corpus, keys=keys, out=encrypted_run, top_k=top_k, threads=threads, timings=timings)
        plain(docs=docs, queries=queries, out=plain_run, top_k=top_k, max_queries=max_queries, timings=timings)
    if encrypted_run.read_bytes() != plain_run.read_bytes():
        raise RunMismatchError(f"{encrypted_run} differs from {plain_run}")
    print("runs match")


def evaluate_runs(*, qrels: Path, runs: Sequence[Path]) -> None:
    named_runs = {f"{path.parent.name}/{path.name}": read_run(path) for path in runs}
    print(format_table(evaluate(read_qrels(qrels), named_runs)), end="")
