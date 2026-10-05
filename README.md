# PrivateRetrieval

PrivateRetrieval ranks an encrypted document corpus against encrypted queries on an untrusted server. The client
embeds documents and queries, quantizes the embeddings to int8, encrypts the documents and turns each query into a
functional key with the function-hiding inner-product scheme `IPFE::OPT` from
[LibPFE](https://github.com/WeiqiNs/LibPFE) on BLS12-381. The server decrypts only the inner product of each
(document, query) pair, ranks the documents itself and returns the top-k as a TREC run. That run is byte-identical to
a plaintext integer search over the same int8 vectors, which the driver checks on every run.

The evaluation uses [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact) with
[Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). The model was chosen because it is
trained for Matryoshka truncation, so a slice of its embedding, renormalized, still ranks well at 64 to 256
dimensions, where encryption is affordable, and because transformers and sentence-transformers support it natively,
without remote code.

## How it works

| Step | Who | Command | Output |
| --- | --- | --- | --- |
| Embed, truncate, quantize | client | `scripts/embed.py` | `docs.i8v`, `queries.i8v`, a float baseline run |
| Master key | client | `privret setup` | `msk.bin`, mode 600 |
| Encrypt the corpus | client | `privret encrypt` | `corpus.ct` |
| Query keys | client | `privret keygen` | `queries.sk` |
| Score and rank a shard | server | `privret search --shard i/N` | one TREC run per shard |
| Combine shards | server | `privret merge` | the encrypted run |
| Reference | anyone with the vectors | `privret plain` | the plaintext integer run |

Each embedding is sliced to the target dimension, L2-normalized and scaled by one factor per dimension,
`s = 127 / max |doc entry|`, fixed from the documents at index time and reused for queries, whose entries are
clipped to [-127, 127]; `quant.json` records the scale and the clip count. The ciphertext header stores the largest
squared document norm Nd and the key header the largest squared query norm Nq, so the server bounds every score by
⌊√(Nd·Nq)⌋ (Cauchy–Schwarz) and sizes its discrete-log table from the files alone.

RELIC is single-threaded, so `scripts/search.sh` runs N `search` processes over disjoint document shards and merges
their runs; each `search` process prepares a query's key once (LibRBP's `PreparedG2`) and scores its whole shard
against it. Every byte on disk is defined in `include/privret/format.hpp`; ranking, merging and the score bound live in
`include/privret/search.hpp`.

## Why functional encryption

**The server learns the scores and nothing else.** A functional key for query q lets the server compute ⟨d, q⟩ for
every encrypted document d, and `IPFE::OPT` hides both vectors beyond that value. The server therefore ranks and
returns the top-k on its own, in one non-interactive pass. Homomorphic encryption offers a different capability:
the server learns nothing, not even the scores. Encrypted scores go back to the key holder, so server-side top-k
needs an extra round with the client, client work proportional to the corpus, or expensive comparisons under
encryption. The two are not a like-for-like benchmark, and this repository does not benchmark HE.

**A TEE also lets the server act on the scores**, but it trusts the hardware vendor's attestation and isolation, and
accepts side-channel exposure, instead of a cryptographic assumption. Its cost is close to plaintext, so the `plain`
run serves as a rough stand-in for it. No TEE was measured, because the benchmark machine has no SGX, SEV or TDX.
Against that stand-in, the 16-process encrypted search took about 4×10⁴ to 6×10⁴ times the wall-clock time of the
single-threaded plaintext search over the same pairs (see Results).

| | Plaintext | TEE | HE | This FE system |
| --- | --- | --- | --- | --- |
| Who learns the scores | server | code inside the enclave | only the key holder | server |
| Where top-k runs | server | enclave | client, or server with extra rounds or encrypted comparisons | server |
| Round trips per query batch | 1 | 1 | 1 for scores; more for server-side top-k | 1 |
| Client work | none | attestation | decrypt the scores (proportional to the corpus) or join comparison rounds | key generation per query |
| Trust assumption | trusted server | hardware vendor, no exploitable side channels | lattice hardness | pairing group (generic group model for `IPFE::OPT`) |

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

## Results

Measured on an AMD Ryzen 7 9800X3D (8 cores, 16 threads); embeddings computed on an RTX 5080. Quality comes from
`scripts/evaluate.py` over `data/scifact/qrels.tsv`; timings come from each run's `runs/d<dim>-q30/timing.jsonl`.

**Encrypted runs, 30-query subset** (the first 30 test queries, `--max-queries 30`, top-k 100, 16 search processes).
Quality is on those 30 queries for all three runs.

| dim | float nDCG@10 | int8 nDCG@10 | encrypted nDCG@10 | float R@100 | int8 R@100 | encrypted R@100 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.5664 | 0.5668 | 0.5668 | 0.7833 | 0.7833 | 0.7833 |
| 128 | 0.6263 | 0.6246 | 0.6246 | 0.9000 | 0.9000 | 0.9000 |
| 256 | 0.6711 | 0.6724 | 0.6724 | 0.9333 | 0.9333 | 0.9333 |

| dim | encrypt ms/doc | keygen ms/query | ms/pair per process | wall ms/pair, all processes | parallel search wall s | plaintext search s | ciphertext bytes/doc | key bytes/query |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 3.14 | 9.04 | 25.8 | 1.65 | 257.0 | 0.0057 | 3332 | 6596 |
| 128 | 5.86 | 16.79 | 46.4 | 2.97 | 461.5 | 0.0126 | 6468 | 12804 |
| 256 | 11.64 | 32.78 | 104.1 | 6.66 | 1034.9 | 0.0167 | 12740 | 25220 |

Each run scores 5,183 documents × 30 queries = 155,490 pairs. "ms/pair per process" is the median `ms_per_pair` of
the 16 `search` lines, measured while all 16 processes share 8 cores; "wall ms/pair" divides the `parallel_search`
wall time by all pairs. Encryption and key generation run in one process. Byte counts are per record:
(4 + dim) compressed G1 points of 49 bytes for a document, (4 + dim) G2 points of 97 bytes for a key. Every
`search` process also decodes the full key file and its corpus shard (`load_seconds`, from about 5 s at dim 64 to
17 s at dim 256), which is reported separately from `ms_per_pair`.

**Quantization, all 300 test queries** (plaintext, `privret plain` against the float baseline):

| dim | float nDCG@10 | int8 nDCG@10 | float R@100 | int8 R@100 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.5577 | 0.5572 | 0.8717 | 0.8717 |
| 128 | 0.6277 | 0.6268 | 0.9050 | 0.9083 |
| 256 | 0.6665 | 0.6683 | 0.9300 | 0.9300 |

The encrypted run equals the int8 run byte for byte, so its quality on any query set is the int8 quality.

## Build

Needs CMake 3.25 or newer, a C++20 compiler and GMP. CMake fetches LibPFE, which fetches LibRBP, which builds RELIC;
`RBP_CURVES` defaults to `bls12_381` here, so only that curve is built. GoogleTest is fetched when not installed.

```sh
cmake -B build -S .
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

To build against local checkouts instead of GitHub:

```sh
cmake -B build -S . -DFETCHCONTENT_SOURCE_DIR_PFE=/path/to/LibPFE -DFETCHCONTENT_SOURCE_DIR_RBP=/path/to/LibRBP
```

`ctest` runs the format and search unit tests and a smoke test that drives `scripts/search.sh` end to end on a tiny
fixture.

## Run

The Python side needs torch, sentence-transformers, datasets, numpy and pytrec-eval-terrier, for example in a conda
environment:

```sh
conda create -n privret python=3.12
conda run -n privret pip install torch sentence-transformers datasets numpy pytrec-eval-terrier
```

Export the vectors, the qrels and the float baseline for each dimension, then run the encrypted pipeline and score it:

```sh
conda run -n privret python scripts/embed.py --out data/scifact --dims 64 128 256
scripts/search.sh --bin build/privret --data data/scifact/d64 --out runs/d64-q30 \
    --dim 64 --procs 16 --top-k 100 --max-queries 30
conda run -n privret python scripts/evaluate.py --qrels data/scifact/qrels.tsv \
    data/scifact/d64/float.run runs/d64-q30/plain.run runs/d64-q30/encrypted.run
```

`search.sh` writes `msk.bin`, `corpus.ct`, `queries.sk`, the shard runs, `encrypted.run`, `plain.run` and
`timing.jsonl` to its output directory, and prints `runs match` once `encrypted.run` equals `plain.run`. Drop
`--max-queries` to search every query. `evaluate.py` scores every run on the queries they share. Running `privret`
with no arguments prints the usage of each subcommand.

## License

Apache License 2.0; see [LICENSE](LICENSE).
