from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import databento as db


DATASET = "GLBX.MDP3"
SCHEMA = "mbp-1"
DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "L1_mbp1"

# Supported market roots from your existing 12-market universe.
# This script now downloads selected markets only; it will not default to all markets.
SUPPORTED_MARKETS = [
    "CL",
    "ES",
    "GC",
    "HG",
    "NG",
    "NQ",
    "RTY",
    "SI",
    "YM",
    "ZB",
    "ZC",
    "ZN",
]


def parse_yyyy_mm_dd(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected YYYY-MM-DD."
        ) from exc


def default_date_range() -> tuple[date, date]:
    """
    Default to the most recent completed UTC day as exclusive end,
    then go back 365 calendar days.
    """
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=365)
    return start, end


def iter_date_chunks(start: date, end: date, chunk_days: int):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        yield cur, nxt
        cur = nxt


def normalize_markets(markets: list[str] | None) -> list[str]:
    cleaned = [m.strip().upper() for m in (markets or []) if m.strip()]
    if not cleaned:
        raise SystemExit("ERROR: choose one market with --market ES.")

    supported = set(SUPPORTED_MARKETS)
    invalid = sorted(set(cleaned) - supported)
    if invalid:
        raise SystemExit(
            "ERROR: unsupported market root(s): "
            + ", ".join(invalid)
            + "\nSupported markets: "
            + ", ".join(SUPPORTED_MARKETS)
        )

    # Preserve user order but remove duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for market in cleaned:
        if market not in seen:
            deduped.append(market)
            seen.add(market)

    return deduped


def continuous_symbol(root: str, expiry_index: int) -> str:
    # Databento continuous futures format: ES.v.0 = volume-based front month.
    return f"{root}.v.{expiry_index}"


def validate_symbols(
    client: db.Historical,
    symbols: list[str],
    start: date,
    end: date,
) -> None:
    print("VALIDATE symbology...")

    try:
        result = client.symbology.resolve(
            dataset=DATASET,
            symbols=symbols,
            stype_in="continuous",
            stype_out="instrument_id",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )
    except Exception as exc:
        raise SystemExit(
            f"ERROR: symbology validation request failed: {type(exc).__name__}: {exc}"
        ) from exc

    mappings = result.get("result", {})
    not_found = set(result.get("not_found", []))

    bad: list[str] = []
    for sym in symbols:
        if sym in not_found or not mappings.get(sym):
            bad.append(sym)

    if bad:
        print("\nERROR: these continuous symbols did not validate:", file=sys.stderr)
        for sym in bad:
            print(f"  - {sym}", file=sys.stderr)

        print(
            "\nConfirm the market root is correct and included in your Databento subscription.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    partial = result.get("partial", [])
    if partial:
        print("\nWARN: partially resolved symbols:")
        for sym in partial:
            print(f"  - {sym}")

    print("VALIDATE symbology OK")


def estimate_cost(
    client: db.Historical,
    symbols: list[str],
    start: date,
    end: date,
) -> None:
    try:
        cost = client.metadata.get_cost(
            dataset=DATASET,
            symbols=symbols,
            schema=SCHEMA,
            stype_in="continuous",
            start=start.isoformat(),
            end=end.isoformat(),
        )
        print(f"Estimated Databento cost: ${cost:,.6f}")
    except Exception as exc:
        print(f"WARN: cost estimate failed: {type(exc).__name__}: {exc}")


def download_chunk(
    client: db.Historical,
    market: str,
    symbol: str,
    start: date,
    end: date,
    out_root: Path,
    overwrite: bool,
) -> Path:
    # Final layout:
    #   data/L1_mbp1/{market}/{market}_front_mbp1_YYYY-MM-DD_YYYY-MM-DD.dbn.zst
    out_dir = out_root / market
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / (
        f"{market}_front_mbp1_{start.isoformat()}_{end.isoformat()}.dbn.zst"
    )

    if out_file.exists() and out_file.stat().st_size > 0 and not overwrite:
        print(f"SKIP {out_file}")
        return out_file

    # Keep temporary file ending in .dbn.zst so Databento/path handling is unambiguous.
    tmp_file = out_dir / (
        f".tmp_{market}_front_mbp1_{start.isoformat()}_{end.isoformat()}.dbn.zst"
    )

    if tmp_file.exists():
        tmp_file.unlink()

    print(f"DOWNLOAD {market:>3} {symbol:<8} {start.isoformat()} -> {end.isoformat()}")

    try:
        client.timeseries.get_range(
            dataset=DATASET,
            schema=SCHEMA,
            symbols=symbol,
            stype_in="continuous",
            start=start.isoformat(),
            end=end.isoformat(),
            path=str(tmp_file),
        )
    except Exception:
        if tmp_file.exists():
            tmp_file.unlink()
        raise

    if not tmp_file.exists() or tmp_file.stat().st_size <= 0:
        if tmp_file.exists():
            tmp_file.unlink()
        raise RuntimeError(f"empty download file: {tmp_file}")

    tmp_file.replace(out_file)
    return out_file


def write_failures(out_root: Path, failures: list[str]) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    fail_path = out_root / "failures.txt"
    fail_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
    return fail_path


def main() -> None:
    default_start, default_end = default_date_range()

    parser = argparse.ArgumentParser(
        description=(
            "Download Databento CME Globex L1 MBP-1 history for selected "
            "continuous front-month futures."
        )
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_ROOT),
        help=f"Output root directory. Default: {DEFAULT_OUT_ROOT}",
    )

    market_group = parser.add_mutually_exclusive_group(required=True)
    market_group.add_argument(
        "--market",
        help=(
            "Single market root to download, e.g. ES. "
            "Use this for one-by-one downloads."
        ),
    )
    market_group.add_argument(
        "--markets",
        nargs="+",
        help=(
            "Optional multi-market mode, e.g. --markets ES GC NG. "
            "This will still skip existing non-empty files unless --overwrite is set."
        ),
    )

    parser.add_argument(
        "--start",
        type=parse_yyyy_mm_dd,
        default=default_start,
        help=f"Inclusive UTC start date YYYY-MM-DD. Default: {default_start}",
    )
    parser.add_argument(
        "--end",
        type=parse_yyyy_mm_dd,
        default=default_end,
        help=f"Exclusive UTC end date YYYY-MM-DD. Default: {default_end}",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=1,
        help="Download chunk size in days. Default: 1",
    )
    parser.add_argument(
        "--expiry-index",
        type=int,
        default=0,
        help="Continuous expiry index. 0 = front month. Default: 0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing non-empty files.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip symbology validation.",
    )
    parser.add_argument(
        "--dry-run-cost",
        action="store_true",
        help="Validate, estimate cost, then exit without downloading.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Seconds to sleep between requests. Default: 0.25",
    )

    args = parser.parse_args()

    if not os.getenv("DATABENTO_API_KEY"):
        raise SystemExit(
            "ERROR: DATABENTO_API_KEY is not set.\n"
            "PowerShell example:\n"
            '  $env:DATABENTO_API_KEY="db-your-key-here"'
        )

    start: date = args.start
    end: date = args.end

    if start >= end:
        raise SystemExit("ERROR: --start must be before --end.")

    if args.chunk_days < 1:
        raise SystemExit("ERROR: --chunk-days must be >= 1.")

    raw_markets = [args.market] if args.market else args.markets
    markets = normalize_markets(raw_markets)
    symbols = [continuous_symbol(market, args.expiry_index) for market in markets]
    out_root = Path(args.out)

    print(f"Dataset:    {DATASET}")
    print(f"Schema:     {SCHEMA}")
    print(f"Range:      {start.isoformat()} inclusive -> {end.isoformat()} exclusive")
    print(f"Markets:    {', '.join(markets)}")
    print(f"Symbols:    {', '.join(symbols)}")
    print(f"Output:     {out_root.resolve()}")
    print(f"Chunk days: {args.chunk_days}")
    print("")

    client = db.Historical()

    if not args.no_validate:
        validate_symbols(client=client, symbols=symbols, start=start, end=end)

    if args.dry_run_cost:
        estimate_cost(client=client, symbols=symbols, start=start, end=end)
        print("DRY RUN complete. No files downloaded.")
        return

    failures: list[str] = []

    for market, symbol in zip(markets, symbols):
        for chunk_start, chunk_end in iter_date_chunks(
            start=start,
            end=end,
            chunk_days=args.chunk_days,
        ):
            try:
                download_chunk(
                    client=client,
                    market=market,
                    symbol=symbol,
                    start=chunk_start,
                    end=chunk_end,
                    out_root=out_root,
                    overwrite=args.overwrite,
                )
                if args.sleep > 0:
                    time.sleep(args.sleep)
            except Exception as exc:
                msg = (
                    f"{market} {symbol} {chunk_start.isoformat()}->{chunk_end.isoformat()} "
                    f"{type(exc).__name__}: {exc}"
                )
                print(f"FAIL {msg}", file=sys.stderr)
                failures.append(msg)

    if failures:
        fail_path = write_failures(out_root=out_root, failures=failures)
        raise SystemExit(f"DONE with {len(failures)} failures. See {fail_path}")

    print("DONE all downloads completed successfully.")


if __name__ == "__main__":
    main()
