import argparse
from pathlib import Path

import numpy as np

from privret.fileformat import Vectors, encode_vectors

DOCS = Vectors(
    ("d0", "d1", "d2", "d3", "d4", "d5", "d6"),
    np.array(
        [
            [127, 127, 127, 127],
            [1, 2, 3, 4],
            [-5, 0, 5, 10],
            [20, -20, 20, -20],
            [7, 7, 7, 7],
            [0, 0, 0, 1],
            [7, 7, 7, 7],
        ],
        dtype=np.int8,
    ),
)
QUERIES = Vectors(("q0", "q1", "q2"), np.array([[-127, -127, -127, -127], [1, 1, 1, 1], [3, -1, 4, -1]], dtype=np.int8))


def write_fixture(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "docs.i8v").write_bytes(encode_vectors(DOCS))
    (directory / "queries.i8v").write_bytes(encode_vectors(QUERIES))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the tiny docs.i8v and queries.i8v fixture.")
    parser.add_argument("directory", type=Path)
    write_fixture(parser.parse_args().directory)
