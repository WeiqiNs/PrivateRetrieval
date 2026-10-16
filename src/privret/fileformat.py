import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pfe
from pfe.ipfe import opt

from privret.errors import FormatError

type Corpus = Sealed[opt.Ciphertext]
type QueryKeys = Sealed[opt.Key]

MAX_DIM = 4096
INT8_LIMIT = 127
MAX_IDS = 2**32 - 1
VECTORS_MAGIC = b"PRVC"
VECTORS_VERSION = 1
PFE_VERSION = 2
MASTER_KEY_MAGIC = b"PRMK"
ID_BYTES = re.compile(rb"[\x21-\x7e\x80-\xff]{1,65535}")


def check_dim(dim: int) -> None:
    if not 1 <= dim <= MAX_DIM:
        raise FormatError(f"dim {dim} is outside [1, {MAX_DIM}]")


def check_ids(ids: tuple[str, ...]) -> None:
    if not 1 <= len(ids) <= MAX_IDS:
        raise FormatError(f"a table needs between 1 and 2^32 - 1 ids, got {len(ids)}")
    seen: set[str] = set()
    for index, id_ in enumerate(ids):
        if ID_BYTES.fullmatch(id_.encode("utf-8")) is None:
            raise FormatError(f"id {index} must be 1 to 65535 printable non-space bytes")
        if id_ in seen:
            raise FormatError(f"id {index} repeats an earlier id")
        seen.add(id_)


@dataclass(frozen=True, eq=False)
class Vectors:
    ids: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        check_ids(self.ids)
        check_dim(self.dim)
        if (self.values == -INT8_LIMIT - 1).any():
            raise FormatError(f"int8 entries must lie in [-{INT8_LIMIT}, {INT8_LIMIT}]")

    @property
    def dim(self) -> int:
        return self.values.shape[1]

    @property
    def max_norm_sq(self) -> int:
        wide = self.values.astype(np.int64)
        return int((wide * wide).sum(axis=1).max())

    def first(self, count: int) -> "Vectors":
        return Vectors(self.ids[:count], self.values[:count])


class Record(Protocol):
    @property
    def n(self) -> int: ...

    def __bytes__(self) -> bytes: ...


@dataclass(frozen=True)
class SealedKind[R: Record]:
    magic: bytes
    decode: Callable[[bytes], R]
    seal: Callable[[opt.MasterKey, list[int]], R]


CIPHERTEXTS = SealedKind(b"PRCT", opt.Ciphertext.from_bytes, opt.encrypt)
KEYS = SealedKind(b"PRSK", opt.Key.from_bytes, opt.keygen)


@dataclass(frozen=True)
class Sealed[R: Record]:
    kind: SealedKind[R]
    ids: tuple[str, ...]
    dim: int
    max_norm_sq: int
    records: tuple[R, ...]

    def __post_init__(self) -> None:
        for index, record in enumerate(self.records):
            if record.n != self.dim:
                raise FormatError(f"record {index} ({self.ids[index]}) has dim {record.n}, not {self.dim}")


class Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    @property
    def remaining(self) -> int:
        return len(self.data) - self.offset

    def take(self, size: int) -> bytes:
        if size > self.remaining:
            raise FormatError("file is truncated")
        self.offset += size
        return self.data[self.offset - size : self.offset]

    def uint(self, size: int) -> int:
        return int.from_bytes(self.take(size), "little")

    def preamble(self, magic: bytes, version: int) -> int:
        if self.take(len(magic)) != magic:
            raise FormatError(f"expected a {magic.decode()} file")
        found = self.uint(4)
        if found != version:
            raise FormatError(f"unsupported format version {found}")
        dim = self.uint(4)
        check_dim(dim)
        return dim

    def ids(self, count: int) -> tuple[str, ...]:
        ids = tuple(self.id(index) for index in range(count))
        check_ids(ids)
        return ids

    def id(self, index: int) -> str:
        try:
            return self.take(self.uint(2)).decode("utf-8")
        except UnicodeDecodeError as err:
            raise FormatError(f"id {index} is not valid UTF-8") from err


def preamble(magic: bytes, version: int, dim: int) -> bytes:
    return magic + version.to_bytes(4, "little") + dim.to_bytes(4, "little")


def encode_ids(ids: tuple[str, ...]) -> bytes:
    return b"".join(len(raw).to_bytes(2, "little") + raw for raw in (id_.encode("utf-8") for id_ in ids))


def encode_vectors(vectors: Vectors) -> bytes:
    return b"".join(
        [
            preamble(VECTORS_MAGIC, VECTORS_VERSION, vectors.dim),
            len(vectors.ids).to_bytes(4, "little"),
            encode_ids(vectors.ids),
            vectors.values.tobytes(),
        ]
    )


def decode_vectors(data: bytes) -> Vectors:
    reader = Reader(data)
    dim = reader.preamble(VECTORS_MAGIC, VECTORS_VERSION)
    ids = reader.ids(reader.uint(4))
    size = len(ids) * dim
    if reader.remaining != size:
        raise FormatError(f"{len(ids)} records of {dim} bytes need {size} bytes, found {reader.remaining}")
    return Vectors(ids, np.frombuffer(reader.take(size), dtype=np.int8).reshape(len(ids), dim))


def encode_sealed[R](table: Sealed[R]) -> bytes:
    return b"".join(
        [
            preamble(table.kind.magic, PFE_VERSION, table.dim),
            len(table.ids).to_bytes(4, "little"),
            table.max_norm_sq.to_bytes(8, "little"),
            encode_ids(table.ids),
            *(bytes(record) for record in table.records),
        ]
    )


def decode_sealed[R: Record](data: bytes, kind: SealedKind[R], *, threads: int = 1) -> Sealed[R]:
    reader = Reader(data)
    dim = reader.preamble(kind.magic, PFE_VERSION)
    count = reader.uint(4)
    max_norm_sq = reader.uint(8)
    limit = INT8_LIMIT * INT8_LIMIT * dim
    if max_norm_sq > limit:
        raise FormatError(f"max_norm_sq {max_norm_sq} exceeds {limit} for dim {dim}")
    ids = reader.ids(count)
    size, extra = divmod(reader.remaining, count)
    if extra:
        raise FormatError(f"{reader.remaining} bytes of records do not split evenly over {count} ids")
    start = reader.offset

    def decode(index: int) -> R:
        try:
            return kind.decode(data[start + index * size : start + (index + 1) * size])
        except pfe.DecodeError as err:
            raise FormatError(f"record {index} ({ids[index]}) does not decode: {err}") from err

    with ThreadPoolExecutor(threads) as pool:
        records = tuple(pool.map(decode, range(count)))
    return Sealed(kind, ids, dim, max_norm_sq, records)


def encode_master_key(msk: opt.MasterKey) -> bytes:
    return preamble(MASTER_KEY_MAGIC, PFE_VERSION, msk.n) + bytes(msk)


def decode_master_key(data: bytes) -> opt.MasterKey:
    reader = Reader(data)
    dim = reader.preamble(MASTER_KEY_MAGIC, PFE_VERSION)
    try:
        msk = opt.MasterKey.from_bytes(reader.take(reader.remaining))
    except pfe.DecodeError as err:
        raise FormatError(f"the master key does not decode: {err}") from err
    if msk.n != dim:
        raise FormatError(f"the master key has dim {msk.n} but its header says {dim}")
    return msk
