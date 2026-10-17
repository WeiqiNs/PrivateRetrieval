# PrivateRetrieval for agents

`privret` ranks an encrypted embedding corpus against encrypted queries: the client quantizes embeddings to int8,
encrypts documents and turns queries into functional keys; the server decrypts only the scores and ranks. The scheme
is `pfe.ipfe.opt` from RustPFE's Python binding, a git dependency pinned to a RustPFE commit in `pyproject.toml`
`[project] dependencies`; it builds from source, so installing needs a Rust toolchain. `README.md` is the user-facing
overview, threat model and results.

## Commands (as `.github/workflows/ci.yml` runs them)

```bash
python -m pip install --upgrade pip && python -m pip install -e . --group dev   # --group needs pip >= 25.1
ruff check && ruff format --check
mypy                                   # strict over src and tests, per pyproject [tool.mypy]
python -m pytest -q
python tests/fixture.py /tmp/smoke
privret run --data /tmp/smoke --out /tmp/smoke/out --top-k 5 --max-queries 2 --threads 3   # prints "runs match"
```

The `embed` extra (`pip install -e ".[embed]"`) is needed only by `privret embed`, which downloads SciFact and the
model named by `embed.MODEL`; CI and the tests never install or run it, and without it the command exits 1.

## Layout

- `src/privret/cli.py`: `parser` and thin handlers that map args to one `pipeline` call; `main` prints any
  `PrivretError` or `OSError` as `privret: <message>` and returns 1.
- `src/privret/pipeline.py`: all file I/O and timing (`embed`, `setup`, `seal_file`, `search`, `plain`, `run`,
  `evaluate_runs`); `Timings` writes one JSON line per step, to stderr for standalone commands.
- `src/privret/fileformat.py`: `Vectors`, `Sealed`/`SealedKind` (`CIPHERTEXTS`, `KEYS`), `Corpus`/`QueryKeys`, and
  every binary codec with its validation. `errors.py`: `PrivretError` and its subclasses.
- `src/privret/client.py` (`seal`) is the key holder's side; `server.py` (`score_table`, `search_encrypted`) the
  server's. `ranking.py`: `rank`, `search_plain`, `format_run` (TREC lines).
- `src/privret/quantize.py`: `quantize`. `export.py`: `export_dim`, the torch-free per-dimension export.
  `embed.py`: SciFact loading and embedding. `evaluate.py`: qrels/run parsing and nDCG@10, Recall@100.
- `tests/fixture.py`: the tiny vector dataset behind `test_pipeline.py` and the smoke run. `test_fileformat.py`:
  byte-exact encodings, malformed input. `test_search.py`: encrypted vs plain search. `test_pipeline.py`: `privret
  run` via `cli.main`, and `pipeline.embed` output from synthetic embeddings. `test_ranking.py`, `test_evaluate.py`.

## Data flow and files

`embed.embed` -> `export.export_dim` (quantize, float `rank`) -> `pipeline.embed` writes `qrels.tsv` and
`d<dim>/{docs.i8v,queries.i8v,float.run,quant.json}`. Then `pipeline.setup` (`msk.bin`), `seal_file` with
`ENCRYPT`/`KEYGEN` (`corpus.ct`/`queries.sk`), `search` (encrypted run), `plain` (int8 run); `pipeline.run` does all
of these into one directory plus `timing.jsonl`; `evaluate_runs` scores runs against qrels.

Binary files start with `fileformat.preamble` (magic, u32 version, u32 dim, little-endian) and use length-prefixed
ids (`encode_ids`): vectors (`VECTORS_MAGIC`, `VECTORS_VERSION`) hold row-major int8; sealed tables
(`SealedKind.magic`, `PFE_VERSION`) hold a count, u64 `max_norm_sq`, ids and fixed-size pfe records; the master key
(`MASTER_KEY_MAGIC`, `PFE_VERSION`) holds pfe bytes. Decoding goes through `Reader` and pfe `from_bytes`, never pickle.

- New command: a `pipeline` function (I/O plus a `Timings` record) and a handler plus `command(...)` in `cli.parser`.
  Only `cli.py` and `pipeline.py` may `print` (ruff `T201` per-file ignores).
- New format or sealed kind: a codec pair or `SealedKind` in `fileformat.py`, a `Sealing` in `pipeline.py` (timing
  field names), cases in `test_fileformat.py`.
- Scheme change: `pfe.ipfe.opt` appears in `fileformat.py`, `client.py`, `server.py` and `pipeline.setup`;
  `PFE_VERSION` versions every file holding pfe bytes.

## Design facts

- The encrypted run equals the int8 plain run byte for byte; `pipeline.run` raises `RunMismatchError` otherwise
  (`test_search.py::test_encrypted_search_matches_plain_search_exactly`, `test_pipeline.py`).
- `ranking.rank` is the one ordering, score descending then doc id ascending, for float, plain and encrypted runs
  (`test_ranking.py`). Results do not depend on `--threads`; the search test runs 1, 3 and 16 threads over 10 docs.
- `server.score_table` sizes the discrete-log table from headers alone, bounding scores by `isqrt(Nd * Nq)`.
  `decode_sealed` rejects `max_norm_sq` above `127^2 * dim`, so a forged header cannot demand a larger table, and
  `Vectors` rejects -128 so the bound holds (`test_fileformat.py`). A score outside the table means keys from another
  master key: `search_encrypted` raises `DecryptError` (`test_search.py`).
- `server.py` never imports `client.py`; `Corpus`/`QueryKeys` are `Sealed[opt.Ciphertext]`/`Sealed[opt.Key]`, so
  mypy rejects passing one as the other.
- Ids are unique, 1 to 65535 printable non-space UTF-8 bytes (`fileformat.check_ids`), because qrels and run lines
  are whitespace-split (`evaluate.fields`).
- `pipeline.write_private` opens `msk.bin` with mode 0o600 and also `fchmod`s it, because the open mode is ignored
  when the file already exists (`test_pipeline.py` checks the mode on a fresh run).
- `FormatError` covers malformed files (including wrapped `pfe.DecodeError`), dim mismatches (`client.seal`,
  `ranking.search_plain`, `server.search_encrypted`) and bad qrels/run lines.
- `quantize` fixes one scale from the documents (`127 / max |doc entry|`) and reuses it for queries, clipping them;
  `quant.json` records scale and clip count (`test_pipeline.py::test_embed_writes_vectors_float_run_summary_and_qrels`).
- `embed.py` alone imports torch, datasets and sentence-transformers; `cli.embed` imports it lazily (ruff `PLC0415`
  ignore) and maps `ModuleNotFoundError` to `MissingExtraError`; `[[tool.mypy.overrides]]` treats those libraries as
  untyped. `pfe` is type-checked against the `.pyi` stubs it ships.

## CI

One job in `.github/workflows/ci.yml`, on push and PR to `main`, over the Python matrix defined there: stable Rust,
install without `embed`, lint, mypy, pytest, smoke run.
