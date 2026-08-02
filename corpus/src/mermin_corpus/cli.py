"""Corpus command line: status, fetch, probe."""
from __future__ import annotations

import argparse
import sys

from .errors import CorpusError
from .fetch import fetch_entry
from .manifest import Manifest, load
from .probe import probe_entry


def _select(manifest: Manifest, args: argparse.Namespace) -> list[str]:
    if args.id:
        return [args.id]
    if args.rung is not None:
        return [e.id for e in manifest.entries if e.rung == args.rung]
    return [e.id for e in manifest.entries]


def _status(manifest: Manifest) -> int:
    width = max((len(e.id) for e in manifest.entries), default=4)
    for e in sorted(manifest.entries, key=lambda x: (x.rung, x.id)):
        print(
            f"rung {e.rung}  {e.id:<{width}}  {e.partition:<15}  "
            f"{e.provenance.get('status', 'pending'):<8}  "
            f"{e.provenance.get('raw_bytes', 0):>14,} B"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mermin-corpus")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")

    for name in ("fetch", "probe"):
        p = sub.add_parser(name)
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument("--id")
        group.add_argument("--rung", type=int)
        group.add_argument("--all", action="store_true")
        if name == "fetch":
            p.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)

    try:
        manifest = load()
        if args.command == "status":
            return _status(manifest)
        selected = _select(manifest, args)
    except CorpusError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    failed: list[str] = []
    for entry_id in selected:
        try:
            manifest = load()
            if args.command == "fetch":
                path = fetch_entry(manifest, entry_id, force=args.force)
                print(f"fetched {entry_id} -> {path}")
            else:
                info = probe_entry(manifest, entry_id)
                print(f"probed  {entry_id}  {info['axes']} {info['shape']} {info['dtype']}")
        except CorpusError as exc:
            failed.append(entry_id)
            print(f"FAILED  {entry_id}: {exc}", file=sys.stderr)

    if failed:
        print(f"{len(failed)} of {len(selected)} entries failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
