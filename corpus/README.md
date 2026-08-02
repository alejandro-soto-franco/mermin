# mermin ingest corpus

Fetch and probe tooling for the multi-channel microscopy corpus that mermin's
ingest layer is tested against.

## Licence position

No imagery is committed to this repository and none is redistributed, including
derived crops. The corpus holds fetched bytes on ASF-EX1, partitioned by what
each source permits:

- `commercial-safe/` CC-BY, CC0 and equivalent.
- `eval-only/` restricted sources. BBBC021 is copyright AstraZeneca Pharmaceuticals.
- `private/` collaborator data, registered in place and never copied or shared.

Each entry cites its source. The original deposit is the record.

## Running it

```bash
uv run --project corpus mermin-corpus status
uv run --project corpus mermin-corpus fetch --rung 0
uv run --project corpus mermin-corpus probe --all
```

The corpus root defaults to `/mnt/ASF-EX1/mermin-corpus` and is overridden by
`MERMIN_CORPUS_ROOT`. Any root under `/mnt/` is mount-asserted before a write,
because the mount point is an ordinary directory on the root volume when the
drive is absent.

## Tests

```bash
uv run --project corpus pytest corpus/tests -v
```

These need no network and no drive. Tests marked `drive` skip when ASF-EX1 is
not mounted.
