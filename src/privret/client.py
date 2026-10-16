from pfe.ipfe import opt

from privret.errors import FormatError
from privret.fileformat import Record, Sealed, SealedKind, Vectors


def seal[R: Record](msk: opt.MasterKey, vectors: Vectors, kind: SealedKind[R]) -> Sealed[R]:
    if vectors.dim != msk.n:
        raise FormatError(f"vectors have dim {vectors.dim} but the master key has dim {msk.n}")
    records = tuple(kind.seal(msk, row.tolist()) for row in vectors.values)
    return Sealed(kind, vectors.ids, vectors.dim, vectors.max_norm_sq, records)
