from collections.abc import Callable

import numpy as np
import pfe
import pytest
from pfe.ipfe import opt

from privret.client import seal
from privret.errors import FormatError
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
    encode_vectors,
)

SAMPLE = Vectors(("a", "b"), np.array([[3, 4, 0], [-127, 0, 1]], dtype=np.int8))
SAMPLE_BYTES = bytes.fromhex("50525643 01000000 03000000 02000000 0100 61 0100 62 030400 810001")


def put(offset: int, value: bytes) -> Callable[[bytearray], None]:
    def mutate(data: bytearray) -> None:
        data[offset : offset + len(value)] = value

    return mutate


def test_vectors_round_trip_and_derive_the_norm() -> None:
    assert SAMPLE.max_norm_sq == 16130
    assert encode_vectors(SAMPLE) == SAMPLE_BYTES
    decoded = decode_vectors(SAMPLE_BYTES)
    assert decoded.ids == ("a", "b")
    assert decoded.values.tolist() == [[3, 4, 0], [-127, 0, 1]]
    assert SAMPLE.first(1).max_norm_sq == 25


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (put(3, b"X"), "expected a PRVC file"),
        (put(4, b"\x02"), "unsupported format version 2"),
        (put(8, b"\x00"), "dim 0 is outside"),
        (put(8, b"\x01\x10"), "dim 4097 is outside"),
        (put(12, b"\x00"), "between 1 and 2\\^32 - 1 ids, got 0"),
        (lambda data: data.pop(), "need 6 bytes, found 5"),
        (lambda data: data.append(0), "need 6 bytes, found 7"),
        (put(26, b"\x80"), "int8 entries must lie in"),
        (put(21, b"a"), "id 1 repeats an earlier id"),
        (put(21, b" "), "id 1 must be 1 to 65535 printable"),
        (put(19, b"\x00"), "id 1 must be 1 to 65535 printable"),
        (put(21, b"\xff"), "id 1 is not valid UTF-8"),
    ],
    ids=[
        "magic",
        "version",
        "dim 0",
        "dim 4097",
        "count 0",
        "truncated",
        "trailing byte",
        "entry -128",
        "duplicate id",
        "id with a space",
        "empty id",
        "invalid UTF-8 id",
    ],
)
def test_rejects_malformed_vectors(mutate: Callable[[bytearray], None], message: str) -> None:
    data = bytearray(SAMPLE_BYTES)
    mutate(data)
    with pytest.raises(FormatError, match=message):
        decode_vectors(bytes(data))


def test_rejects_vectors_where_ciphertexts_are_expected() -> None:
    with pytest.raises(FormatError, match="expected a PRCT file"):
        decode_sealed(SAMPLE_BYTES, CIPHERTEXTS)


@pytest.mark.parametrize("kind", [CIPHERTEXTS, KEYS], ids=["ciphertexts", "keys"])
def test_sealed_tables_round_trip(kind: SealedKind[Record]) -> None:
    sealed = seal(opt.setup(3), SAMPLE, kind)
    records = [bytes(record) for record in sealed.records]
    data = encode_sealed(sealed)
    assert data[30:] == b"".join(records)
    decoded = decode_sealed(data, kind, threads=2)
    assert decoded.ids == ("a", "b")
    assert decoded.max_norm_sq == 16130
    assert [bytes(record) for record in decoded.records] == records


def test_sealed_tables_reject_corruption() -> None:
    data = encode_sealed(seal(opt.setup(3), SAMPLE, CIPHERTEXTS))
    at_limit = bytearray(data)
    at_limit[16:24] = (127 * 127 * 3).to_bytes(8, "little")
    assert decode_sealed(bytes(at_limit), CIPHERTEXTS).max_norm_sq == 127 * 127 * 3
    over_limit = bytearray(data)
    over_limit[16:24] = (127 * 127 * 3 + 1).to_bytes(8, "little")
    with pytest.raises(FormatError, match="max_norm_sq 48388 exceeds 48387"):
        decode_sealed(bytes(over_limit), CIPHERTEXTS)
    version_one = bytearray(data)
    version_one[4] = 1
    with pytest.raises(FormatError, match="unsupported format version 1"):
        decode_sealed(bytes(version_one), CIPHERTEXTS)
    with pytest.raises(FormatError, match="do not split evenly over 2 ids"):
        decode_sealed(data[:-1], CIPHERTEXTS)
    wrong_n = bytearray(data)
    wrong_n[30:34] = (4).to_bytes(4, "big")
    with pytest.raises(FormatError, match=r"record 0 \(a\) does not decode") as info:
        decode_sealed(bytes(wrong_n), CIPHERTEXTS)
    assert isinstance(info.value.__cause__, pfe.DecodeError)


def test_master_key_round_trips() -> None:
    msk = opt.setup(3)
    data = encode_master_key(msk)
    decoded = decode_master_key(data)
    assert bytes(decoded) == bytes(msk)
    with pytest.raises(FormatError, match="the master key does not decode"):
        decode_master_key(data[:-1])
