# PrivateRetrieval for agents

Private embedding retrieval over function-hiding inner-product functional encryption: a Python package, `privret`, on
RustPFE's `pfe` binding (`pfe.ipfe.opt` on BLS12-381). The client embeds and quantizes SciFact to int8, encrypts the
documents and turns queries into functional keys; the server decrypts only the scores and ranks. `README.md` is the
user-facing overview.

## Commands

```bash
pip install -e ".[embed]" --group dev   # drop [embed] when torch is not needed
ruff check && ruff format --check
python -m pytest -q
python tests/fixture.py /tmp/smoke && privret run --data /tmp/smoke --out /tmp/smoke/out --top-k 5 --max-queries 2 --threads 3
```

CI runs the same commands on Python 3.12 and 3.13. `pfe` builds from source, so a Rust toolchain is needed.

## Invariants

- The encrypted run equals the int8 plaintext run byte for byte; `pipeline.run` checks it on every run and raises
  `RunMismatchError` otherwise.
- `ranking.rank` owns the order (score descending, then doc id ascending) for plain, encrypted and float runs.
- The score bound comes from the corpus and key headers alone, via `server.score_table`.
- `server` takes only `Corpus` and `QueryKeys` and never imports `client`.
- `fileformat` owns the vector, corpus, key and master-key formats; files are decoded with `from_bytes`, never
  pickled. Ids are UTF-8.
- Results are identical for any `--threads`, including more threads than documents.
- No comments or docstrings in code.

## Layout

- `src/privret/fileformat.py`: vector, sealed-table and master-key encodings and their validation.
- `src/privret/ranking.py`: `rank`, plaintext search and TREC formatting.
- `src/privret/client.py`, `server.py`: sealing on the key holder's side; scoring and ranking on the server's.
- `src/privret/pipeline.py`: every command's file I/O and timing; `cli.py` only parses arguments.
- `src/privret/embed.py`, `quantize.py`, `evaluate.py`: embedding (the `embed` extra), int8 quantization, quality.
- `tests/`: `test_search.py` checks encrypted against plain search, `test_pipeline.py` drives the CLI end to end on
  `fixture.py`, the tiny dataset the CI smoke run also uses.
