# Private Embedding Retrieval with Functional Encryption (PrivateRetrieval)

[![PrivateRetrieval CI](https://github.com/WeiqiNs/PrivateRetrieval/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/WeiqiNs/PrivateRetrieval/actions/workflows/ci.yml)

PrivateRetrieval ranks an encrypted document corpus against encrypted queries on an untrusted server. The client
embeds documents and queries, quantizes the embeddings to int8, encrypts the documents and turns each query into a
functional key with the function-hiding inner-product scheme of Ojaswi et al. (`pfe.ipfe.opt` from
[RustPFE](https://github.com/WeiqiNs/RustPFE)'s Python binding, on BLS12-381). The server decrypts only the inner
product of each (document, query) pair, ranks the documents itself and returns the top-k as a TREC run. That run is
byte-identical to a plaintext integer search over the same int8 vectors, which `privret run` checks every time.

The evaluation uses [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact) with
[Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). The model was chosen because it is
trained for Matryoshka truncation, so a slice of its embedding, renormalized, still ranks well at 64 to 256
dimensions, where encryption is affordable, and because transformers and sentence-transformers support it natively,
without remote code.

## Quick start

```sh
pip install -e ".[embed]"
privret embed --out data/scifact --dims 64
privret run --data data/scifact/d64 --out runs/d64-q30 --top-k 100 --max-queries 30
privret evaluate --qrels data/scifact/qrels.tsv \
    data/scifact/d64/float.run runs/d64-q30/plain.run runs/d64-q30/encrypted.run
```

`privret run` writes the master key, the encrypted corpus, the query keys, both runs and `timing.jsonl` to its output
directory and prints `runs match` once the encrypted run equals the plaintext one. See [Install](#install) for the
environment.

## Why functional encryption

**The server learns the scores and nothing else.** A functional key for query q lets the server compute ⟨d, q⟩ for
every encrypted document d, and `pfe.ipfe.opt` hides both vectors beyond that value. The server therefore ranks and
returns the top-k on its own, in one non-interactive pass. Homomorphic encryption offers a different capability:
the server learns nothing, not even the scores. Encrypted scores go back to the key holder, so server-side top-k
needs an extra round with the client, client work proportional to the corpus, or expensive comparisons under
encryption. The two are not a like-for-like benchmark, and this repository does not benchmark HE.

**A TEE also lets the server act on the scores**, but it trusts the hardware vendor's attestation and isolation, and
accepts side-channel exposure, instead of a cryptographic assumption. Its cost is close to plaintext, so the `plain`
run serves as a rough stand-in for it. No TEE was measured, because the benchmark machine has no SGX, SEV or TDX.
Against that stand-in, the 16-thread encrypted search took about 1.5×10⁴ to 2.5×10⁴ times the wall-clock time of the plaintext search
over the same pairs (see Results).

| | Plaintext | TEE | HE | This FE system |
| --- | --- | --- | --- | --- |
| Who learns the scores | server | code inside the enclave | only the key holder | server |
| Where top-k runs | server | enclave | client, or server with extra rounds or encrypted comparisons | server |
| Round trips per query batch | 1 | 1 | 1 for scores; more for server-side top-k | 1 |
| Client work | none | attestation | decrypt the scores (proportional to the corpus) or join comparison rounds | key generation per query |
| Trust assumption | trusted server | hardware vendor, no exploitable side channels | lattice hardness | pairing group (generic group model for `pfe.ipfe.opt`) |

### What the server learns

- **Every score.** The server sees the integer ⟨d, q⟩ for every pair it evaluates, so it knows each query's full
  ranking of the corpus, not only the top-k it returns.
- **Sizes and identifiers.** The number of documents and queries, the dimension, the document and query ids (stored
  in the clear), and the norm bounds Nd and Nq from the file headers.
- **Repeated queries.** Two keys for the same query share no bytes, but their score vectors over the corpus are
  identical, which links them.
- **Known plaintexts.** Scores are linear in the hidden vectors. A server that knows the int8 vectors of at least
  `dim` linearly independent documents in the corpus solves a linear system for any query from that query's scores;
  symmetrically, `dim` known queries reveal any document.

## How it works

| Step | Who | Command | Output |
| --- | --- | --- | --- |
| Embed, truncate, quantize | client | `privret embed` | `docs.i8v`, `queries.i8v`, `qrels.tsv`, a float baseline run |
| Master key | client | `privret setup` | `msk.bin`, mode 600 |
| Encrypt the corpus | client | `privret encrypt` | `corpus.ct` |
| Query keys | client | `privret keygen` | `queries.sk` |
| Score and rank | server | `privret search --threads N` | the encrypted run |
| Reference | anyone with the vectors | `privret plain` | the plaintext integer run |
| Everything above, checked | client | `privret run` | all of the above, `timing.jsonl`, and `runs match` |
| Quality | anyone with the qrels | `privret evaluate` | nDCG@10 and Recall@100 per run |

Each embedding is sliced to the target dimension, L2-normalized and scaled by one factor per dimension,
`s = 127 / max |doc entry|`, fixed from the documents at index time and reused for queries, whose entries are
clipped to [-127, 127]; `quant.json` records the scale and the clip count. The ciphertext header stores the largest
squared document norm Nd and the key header the largest squared query norm Nq, so the server bounds every score by
⌊√(Nd·Nq)⌋ (Cauchy–Schwarz) and sizes its discrete-log table from the files alone.

`search` runs in one process. It decodes the corpus and the keys on a pool of `--threads` Python threads, prepares
each query's key once (`opt.prepare`, which precomputes the key's side of every pairing), and decrypts the whole corpus
against it on a pool of the same size. The binding releases the GIL while it decodes and decrypts, so the threads run in
parallel, and RustPFE's rayon pool spreads each decryption's pairing product over the cores as well. The ranking is
the same for any thread count. The vector, corpus, key and master-key file formats are defined in
`src/privret/fileformat.py`; ranking lives in `ranking.rank`.

## Results

Measured on an AMD Ryzen 7 9800X3D (8 cores, 16 threads) with Python 3.12; embeddings computed on an RTX 5080. Quality
comes from `privret evaluate` over `data/scifact/qrels.tsv`; timings come from each run's `timing.jsonl`.

**Encrypted runs, 30-query subset** (the first 30 test queries, `--max-queries 30`, top-k 100, `--threads 16`).
Quality is on those 30 queries for all three runs.

| dim | float nDCG@10 | int8 nDCG@10 | encrypted nDCG@10 | float R@100 | int8 R@100 | encrypted R@100 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.5664 | 0.5668 | 0.5668 | 0.7833 | 0.7833 | 0.7833 |
| 128 | 0.6263 | 0.6246 | 0.6246 | 0.9000 | 0.9000 | 0.9000 |
| 256 | 0.6711 | 0.6724 | 0.6724 | 0.9333 | 0.9333 | 0.9333 |

| dim | encrypt ms/doc | keygen ms/query | ms/pair, one core | ms/pair, 16 threads | search wall s | plaintext search s | search ÷ plaintext | ciphertext bytes/doc | key bytes/query |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.178 | 0.446 | 10.0 | 1.09 | 169.7 | 0.0112 | 1.5×10⁴ | 3268 | 6532 |
| 128 | 0.270 | 0.676 | 18.4 | 1.98 | 308.6 | 0.0151 | 2.0×10⁴ | 6340 | 12676 |
| 256 | 0.425 | 1.101 | 39.0 | 3.74 | 581.8 | 0.0230 | 2.5×10⁴ | 12484 | 24964 |

Each run scores 5,183 documents × 30 queries = 155,490 pairs. "ms/pair, 16 threads" is the search's `ms_per_pair`, its
wall time divided by all pairs; "one core" is the `ms_per_pair` of a one-query run with `--threads 1` and
`RAYON_NUM_THREADS=1`. The plaintext search is one numpy matrix product plus the same ranking, on one thread.
Encryption and key generation call RustPFE once per vector from one Python thread. Byte counts are per record: the
4-byte dim, then (4 + dim) compressed G1 points of 48 bytes for a document or (4 + dim) G2 points of 96 bytes for a
key. Before scoring, `search` decodes the corpus and the keys (`load_seconds`, from 1.8 s at dim 64 to 6.8 s at dim 256
on 16 threads), which is reported separately from `ms_per_pair`.

**Quantization, all 300 test queries** (plaintext, `privret plain` against the float baseline):

| dim | float nDCG@10 | int8 nDCG@10 | float R@100 | int8 R@100 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.5577 | 0.5572 | 0.8717 | 0.8717 |
| 128 | 0.6277 | 0.6268 | 0.9050 | 0.9083 |
| 256 | 0.6665 | 0.6683 | 0.9300 | 0.9300 |

The encrypted run equals the int8 run byte for byte, so its quality on any query set is the int8 quality.

## Install

Python 3.12 or newer and a Rust toolchain ([rustup](https://rustup.rs)), because `pfe` builds from source. For
example, in a conda environment:

```sh
conda create -n privret python=3.12
conda activate privret
pip install -e ".[embed]" --group dev
```

`--group dev` adds pytest, ruff and mypy and needs pip 25.1 or newer. The `embed` extra pulls in torch,
sentence-transformers and datasets and is only needed for `privret embed`; without it, `pip install -e .` installs
everything else, which is enough to encrypt, search and evaluate existing vector files.

## Reproduce

Export the vectors, the qrels and the float baseline for each dimension, then run the encrypted pipeline and score it:

```sh
privret embed --out data/scifact --dims 64 128 256
for d in 64 128 256; do
    privret run --data data/scifact/d$d --out runs/d$d-q30 --top-k 100 --max-queries 30 --threads 16
    privret plain --docs data/scifact/d$d/docs.i8v --queries data/scifact/d$d/queries.i8v --top-k 100 \
        --out runs/d$d/plain.run
done
privret evaluate --qrels data/scifact/qrels.tsv \
    data/scifact/d64/float.run runs/d64-q30/plain.run runs/d64-q30/encrypted.run
```

Drop `--max-queries` to search every query. `--threads` defaults to the number of CPUs. `evaluate` scores every run
on the queries they share. Each subcommand prints its options with `--help`; the standalone commands write their
timing line to stderr.

## Tests

```sh
ruff check && ruff format --check
mypy
python -m pytest -q
```

The tests cover the file formats and their rejection of malformed input, the ranking order, encrypted search against
plaintext search for several thread counts, quantization, evaluation, and an end-to-end `privret run` on a tiny
fixture (`tests/fixture.py`), which CI also runs as a smoke test on Python 3.12 and 3.13.

## The PFE family

LibPFE, GoPFE and RustPFE implement the same eight inner-product and quadratic functional encryption schemes with the
same test cases and benchmark inputs; LibRBP is the pairing layer under LibPFE. PrivateRetrieval is an application
built on RustPFE's Python binding.

| Repository | Language | Pairing backend | Role |
| --- | --- | --- | --- |
| [LibRBP](https://github.com/WeiqiNs/LibRBP) | C++20 | [RELIC](https://github.com/relic-toolkit/relic) | Typed, thread-safe pairing library: curves, groups, pairings, linear algebra, dlog tables |
| [LibPFE](https://github.com/WeiqiNs/LibPFE) | C++20 | LibRBP | The eight schemes as header-only templates over any LibRBP curve |
| [GoPFE](https://github.com/WeiqiNs/GoPFE) | Go | [gnark-crypto](https://github.com/Consensys/gnark-crypto) | The eight schemes on BLS12-381, parallel across `GOMAXPROCS` |
| [RustPFE](https://github.com/WeiqiNs/RustPFE) | Rust, Python | [arkworks](https://github.com/arkworks-rs/algebra) | The eight schemes in safe Rust over any arkworks pairing curve, with a PyO3 Python binding |
| **PrivateRetrieval** (this repo) | Python | RustPFE | Private embedding retrieval on `pfe.ipfe.opt` |

## License

Apache License 2.0; see [LICENSE](LICENSE).
