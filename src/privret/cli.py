import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from privret import pipeline
from privret.errors import MissingExtraError, PrivretError
from privret.fileformat import MAX_DIM
from privret.pipeline import ENCRYPT, KEYGEN, Timings


def positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"'{text}' is not a positive integer")
    return value


def dimension(text: str) -> int:
    value = positive(text)
    if value > MAX_DIM:
        raise argparse.ArgumentTypeError(f"must be at most {MAX_DIM}")
    return value


def stderr_timings() -> Timings:
    return Timings(sys.stderr)


def setup(args: argparse.Namespace) -> None:
    pipeline.setup(dim=args.dim, msk=args.msk, timings=stderr_timings())


def encrypt(args: argparse.Namespace) -> None:
    pipeline.seal_file(
        sealing=ENCRYPT, msk=args.msk, vectors=args.vectors, out=args.out, limit=None, timings=stderr_timings()
    )


def keygen(args: argparse.Namespace) -> None:
    pipeline.seal_file(
        sealing=KEYGEN,
        msk=args.msk,
        vectors=args.vectors,
        out=args.out,
        limit=args.max_queries,
        timings=stderr_timings(),
    )


def search(args: argparse.Namespace) -> None:
    pipeline.search(
        corpus=args.corpus,
        keys=args.keys,
        out=args.out,
        top_k=args.top_k,
        threads=args.threads,
        timings=stderr_timings(),
    )


def plain(args: argparse.Namespace) -> None:
    pipeline.plain(
        docs=args.docs,
        queries=args.queries,
        out=args.out,
        top_k=args.top_k,
        max_queries=args.max_queries,
        timings=stderr_timings(),
    )


def run(args: argparse.Namespace) -> None:
    pipeline.run(data=args.data, out=args.out, top_k=args.top_k, max_queries=args.max_queries, threads=args.threads)


def embed(args: argparse.Namespace) -> None:
    try:
        from privret import embed as embedding
    except ModuleNotFoundError as err:
        raise MissingExtraError(
            f"privret embed needs the embed extra (missing {err.name}): pip install -e '.[embed]'"
        ) from err
    pipeline.embed(export=embedding.embed(dims=args.dims, batch_size=args.batch_size, device=args.device), out=args.out)


def evaluate(args: argparse.Namespace) -> None:
    pipeline.evaluate_runs(qrels=args.qrels, runs=args.runs)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="privret", description="Private embedding retrieval over IPFE.")
    commands = root.add_subparsers(dest="command", required=True)

    def command(name: str, handler: Callable[[argparse.Namespace], None], help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text)
        sub.set_defaults(handler=handler)
        return sub

    def path(sub: argparse.ArgumentParser, *flags: str) -> None:
        for flag in flags:
            sub.add_argument(flag, type=Path, required=True)

    def top_k(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--top-k", type=positive, required=True)

    def max_queries(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--max-queries", type=positive)

    def threads(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--threads", type=positive, default=os.cpu_count() or 1)

    sub = command("embed", embed, "embed SciFact and write int8 vectors, qrels and a float baseline run")
    path(sub, "--out")
    sub.add_argument("--dims", type=dimension, nargs="+", default=[64, 128, 256])
    sub.add_argument("--batch-size", type=positive, default=32)
    sub.add_argument("--device")

    sub = command("setup", setup, "write a master key")
    sub.add_argument("--dim", type=dimension, required=True)
    path(sub, "--msk")

    sub = command("encrypt", encrypt, "encrypt a document vector file")
    path(sub, "--msk", "--vectors", "--out")

    sub = command("keygen", keygen, "derive functional keys for a query vector file")
    path(sub, "--msk", "--vectors", "--out")
    max_queries(sub)

    sub = command("search", search, "rank an encrypted corpus against query keys")
    path(sub, "--corpus", "--keys")
    top_k(sub)
    path(sub, "--out")
    threads(sub)

    sub = command("plain", plain, "rank plaintext int8 vectors")
    path(sub, "--docs", "--queries")
    top_k(sub)
    path(sub, "--out")
    max_queries(sub)

    sub = command("run", run, "run the whole pipeline and check the runs match")
    path(sub, "--data", "--out")
    top_k(sub)
    max_queries(sub)
    threads(sub)

    sub = command("evaluate", evaluate, "score runs on their shared queries with nDCG@10 and Recall@100")
    path(sub, "--qrels")
    sub.add_argument("runs", type=Path, nargs="+")

    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        args.handler(args)
    except (PrivretError, OSError) as err:
        print(f"privret: {err}", file=sys.stderr)
        return 1
    return 0
