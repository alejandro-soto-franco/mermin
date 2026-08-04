"""Corpus command line: status, fetch, probe."""
from __future__ import annotations

import argparse
import sys

from .errors import CorpusError
from .fetch import fetch_entry
from .generate import GOLDEN_ENTRIES, check_entry, generate_entry
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


def _goldens(args: argparse.Namespace) -> int:
    selected = [args.entry] if args.entry else sorted(GOLDEN_ENTRIES)
    failed: list[str] = []
    for entry_id in selected:
        try:
            manifest = load()
            if args.check:
                diffs = check_entry(manifest, entry_id)
                real = [d for d in diffs if d.kind != "environment"]
                environment = [d for d in diffs if d.kind == "environment"]
                if real:
                    failed.append(entry_id)
                    print(f"FAILED  {entry_id}: {len(real)} difference(s)", file=sys.stderr)
                    for d in real:
                        print(f"  {d}", file=sys.stderr)
                else:
                    print(f"ok      {entry_id}")
                    for d in environment:
                        print(f"  {d}")
            else:
                path = generate_entry(manifest, entry_id)
                print(f"wrote   {entry_id} -> {path}")
        except CorpusError as exc:
            failed.append(entry_id)
            print(f"FAILED  {entry_id}: {exc}", file=sys.stderr)

    if failed:
        print(f"{len(failed)} of {len(selected)} entries failed: {', '.join(failed)}", file=sys.stderr)
        return 1
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

    goldens_parser = sub.add_parser("goldens")
    goldens_parser.add_argument("--entry")
    goldens_parser.add_argument("--check", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "goldens":
        return _goldens(args)

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
