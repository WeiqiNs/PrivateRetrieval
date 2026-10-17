# PrivateRetrieval

[![PrivateRetrieval CI](https://github.com/WeiqiNs/PrivateRetrieval/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/WeiqiNs/PrivateRetrieval/actions/workflows/ci.yml)

Private embedding search: an untrusted server ranks an encrypted document corpus against encrypted queries and
learns only the similarity scores. Documents are encrypted and each query becomes a functional key under the
function-hiding inner-product scheme of Ojaswi et al. (`pfe.ipfe.opt` from [RustPFE](https://github.com/WeiqiNs/RustPFE), BLS12-381), so
the server computes ⟨document, query⟩ for every pair, ranks on its own and returns the top-k in one non-interactive
pass. The encrypted ranking is byte-identical to a plaintext integer search over the same vectors, and every run
checks it.

On [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact) with
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) truncated to 128 dimensions, encrypted search
keeps the int8 baseline's nDCG@10 of 0.625 and scores a document–query pair in about 2 ms on 16 threads.

## Quick start

Needs Python 3.12+ and a Rust toolchain ([rustup](https://rustup.rs)), since `pfe` builds from source.

```sh
pip install -e ".[embed]"
privret embed --out data/scifact --dims 64
privret run --data data/scifact/d64 --out runs/d64-q30 --top-k 100 --max-queries 30
privret evaluate --qrels data/scifact/qrels.tsv \
    data/scifact/d64/float.run runs/d64-q30/plain.run runs/d64-q30/encrypted.run
```

`privret run` sets up a master key, encrypts the corpus, derives the query keys, searches, and prints `runs match` once
the encrypted run equals the plaintext one. The `embed` extra (torch, sentence-transformers, datasets) is only needed
for `privret embed`.

## How it works

| Step | Who | Command |
| --- | --- | --- |
| Embed, truncate to the target dimension, quantize to int8 | client | `privret embed` |
| Master key, encrypted corpus, query keys | client | `privret setup`, `encrypt`, `keygen` |
| Score and rank | server | `privret search --threads N` |
| Plaintext reference, quality | anyone with the vectors | `privret plain`, `evaluate` |

Qwen3-Embedding is trained for Matryoshka truncation, so a renormalized 64–256 dimension slice still ranks well, which
keeps encryption affordable. The file headers carry the largest squared document and query norms, so the server
bounds every score by Cauchy–Schwarz and sizes its discrete-log table from the files alone.

## What the server learns

- **Every score**, so each query's full ranking of the corpus, not only the top-k.
- **Sizes and ids**: document and query counts, the dimension, the ids and the norm bounds.
- **Repeated queries**: two keys for the same query share no bytes but produce identical score vectors.
- **Linear leakage**: knowing the vectors of `dim` linearly independent documents reveals any query from its scores,
  and vice versa.

Homomorphic encryption would hide the scores too, but then server-side top-k needs extra rounds or encrypted
comparisons; a TEE runs at plaintext speed but trusts the hardware vendor. This repository benchmarks neither.

## Results

AMD Ryzen 7 9800X3D (8 cores, 16 threads), 5,183 documents × the first 30 test queries, top-k 100, `--threads 16`:

| dim | int8 / encrypted nDCG@10 | encrypted R@100 | ms per pair | encrypt ms/doc | ciphertext bytes/doc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.5668 | 0.7833 | 1.09 | 0.18 | 3268 |
| 128 | 0.6246 | 0.9000 | 1.98 | 0.27 | 6340 |
| 256 | 0.6724 | 0.9333 | 3.74 | 0.43 | 12484 |

Encrypted search takes 1.5×10⁴ to 2.5×10⁴ times as long as the plaintext numpy search. Int8 quantization costs
at most 0.001 nDCG@10 against the float embeddings over all 300 test queries.

## Reproduce and test

```sh
privret embed --out data/scifact --dims 64 128 256
for d in 64 128 256; do
    privret run --data data/scifact/d$d --out runs/d$d-q30 --top-k 100 --max-queries 30 --threads 16
done
pip install -e . --group dev
ruff check && ruff format --check && mypy && python -m pytest -q
```

Drop `--max-queries` to search every query; each subcommand documents its options under `--help`. The tests cover the
file formats, ranking, encrypted-versus-plaintext search across thread counts, quantization, evaluation and an
end-to-end `privret run`, which CI also runs on Python 3.12 and 3.13.
