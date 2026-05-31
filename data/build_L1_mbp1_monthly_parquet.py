from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import databento as db
except ImportError:
    db = None


DAILY_RE = re.compile(
    r"^(?P<market>[A-Z0-9]+)_front_mbp1_"
    r"(?P<start>\d{4}-\d{2}-\d{2})_"
    r"(?P<end>\d{4}-\d{2}-\d{2})\.dbn\.zst$"
)


@dataclass(frozen=True)
class DailyFile:
    market: str
    start: date
    end: date
    path: Path

    @property
    def month_key(self) -> tuple[str, int, int]:
        return self.market, self.start.year, self.start.month


def parse_daily(path: Path) -> DailyFile | None:
    m = DAILY_RE.match(path.name)
    if not m:
        return None
    return DailyFile(
        market=m.group("market"),
        start=datetime.strptime(m.group("start"), "%Y-%m-%d").date(),
        end=datetime.strptime(m.group("end"), "%Y-%m-%d").date(),
        path=path,
    )


def find_daily_files(root: Path) -> list[DailyFile]:
    files = []
    for path in sorted(root.rglob("*.dbn.zst")):
        if path.name.startswith(".tmp_"):
            continue
        parsed = parse_daily(path)
        if parsed is None:
            continue
        if path.relative_to(root).parts[0] != parsed.market:
            raise SystemExit(f"ERROR: market folder mismatch: {path}")
        files.append(parsed)
    return files


def missing_starts(files: list[DailyFile]) -> list[date]:
    starts = {f.start for f in files}
    cur = min(starts)
    end = max(starts)
    missing = []
    while cur <= end:
        if cur not in starts:
            missing.append(cur)
        cur += timedelta(days=1)
    return missing


def decode_daily(path: Path) -> pd.DataFrame:
    if db is None:
        raise RuntimeError("databento is not installed")
    df = db.DBNStore.from_file(str(path)).to_df()
    if "ts_event" in df.columns:
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df = df.sort_values("ts_event").reset_index(drop=True)
    return df


def write_monthly_parquet(
    files: list[DailyFile],
    out_file: Path,
    overwrite: bool,
    compression: str,
) -> dict:
    if out_file.exists() and out_file.stat().st_size > 0 and not overwrite:
        print(f"SKIP existing {out_file}")
        return {
            "market": out_file.parent.name,
            "file": out_file.name,
            "status": "skipped_existing",
            "daily_files": len(files),
            "rows": None,
            "first_ts": None,
            "last_ts": None,
            "empty_daily_files": None,
        }

    tmp = out_file.with_name(f".tmp_{out_file.name}")
    if tmp.exists():
        tmp.unlink()

    writer = None
    rows = 0
    first_ts = None
    last_ts = None
    empty_daily_files = 0

    try:
        for item in sorted(files, key=lambda x: x.start):
            print(f"READ  {item.path}")
            df = decode_daily(item.path)
            if df.empty:
                empty_daily_files += 1
                print(f"WARN empty decoded daily file: {item.path}")
                continue
            if "ts_event" in df.columns and not df.empty:
                lo = df["ts_event"].min()
                hi = df["ts_event"].max()
                first_ts = lo if first_ts is None else min(first_ts, lo)
                last_ts = hi if last_ts is None else max(last_ts, hi)

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression=compression)
            writer.write_table(table)
            rows += len(df)

        if writer is None or rows == 0:
            raise RuntimeError(f"no rows written for {out_file}")
    finally:
        if writer is not None:
            writer.close()

    if tmp.stat().st_size <= 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"empty parquet output: {out_file}")

    tmp.replace(out_file)
    print(f"WROTE {out_file} daily_files={len(files)} empty_daily_files={empty_daily_files} rows={rows:,} bytes={out_file.stat().st_size:,}")
    return {
        "market": out_file.parent.name,
        "file": out_file.name,
        "status": "written",
        "daily_files": len(files),
        "rows": rows,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "empty_daily_files": empty_daily_files,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build monthly L1 MBP-1 parquet directly from daily .dbn.zst files."
    )
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--markets", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--max-months", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    markets = {m.upper() for m in args.markets} if args.markets else None

    daily = find_daily_files(root)
    if markets:
        daily = [x for x in daily if x.market in markets]
    if not daily:
        raise SystemExit(f"ERROR: no daily .dbn.zst files found under {root}\\{{market}}")

    grouped: dict[tuple[str, int, int], list[DailyFile]] = defaultdict(list)
    for item in daily:
        grouped[item.month_key].append(item)

    groups = sorted(grouped.items())
    if args.max_months > 0:
        groups = groups[: args.max_months]

    manifest = []
    failures = []

    for (market, year, month), files in groups:
        out_file = root / market / f"{year}-{month:02d}.parquet"
        missing = missing_starts(files)
        if missing:
            print(
                f"WARN {market} {year}-{month:02d}: missing daily starts "
                f"{', '.join(d.isoformat() for d in missing[:10])}"
                f"{' ...' if len(missing) > 10 else ''}"
            )
        try:
            if args.dry_run:
                print(f"DRY {out_file} daily_files={len(files)}")
                manifest.append(
                    {
                        "market": market,
                        "file": out_file.name,
                        "status": "dry_run",
                        "daily_files": len(files),
                        "rows": None,
                        "first_ts": None,
                        "last_ts": None,
                        "empty_daily_files": None,
                    }
                )
            else:
                manifest.append(
                    write_monthly_parquet(
                        files=files,
                        out_file=out_file,
                        overwrite=args.overwrite,
                        compression=args.compression,
                    )
                )
        except Exception as exc:
            msg = f"{market} {year}-{month:02d}: {type(exc).__name__}: {exc}"
            print(f"FAIL {msg}")
            failures.append(msg)

    manifest_path = root / "monthly_parquet_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f"WROTE manifest {manifest_path}")

    if failures:
        fail_path = root / "monthly_parquet_failures.txt"
        fail_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise SystemExit(f"DONE with {len(failures)} failures. See {fail_path}")

    print(f"DONE months={len(groups)} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
