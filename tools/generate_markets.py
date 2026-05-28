"""Generate 12 per-market YAML configs from a single spec dict.
Run:  python tools/generate_markets.py
Output: configs/markets/{TICKER}.yaml (12 files, each <= 18 lines)
"""
import yaml
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "configs" / "markets"

# CME standard specs (tick_value = tick_size * multiplier)
MARKETS = {
    "ES":  {"multiplier": 50,     "tick_size": 0.25,      "slippage_k": 0.0005, "vol_penalty": 0.002,  "max_leverage": 3.0, "max_position": 50},
    "NQ":  {"multiplier": 20,     "tick_size": 0.25,      "slippage_k": 0.0005, "vol_penalty": 0.003,  "max_leverage": 2.5, "max_position": 30},
    "YM":  {"multiplier": 5,      "tick_size": 1.0,       "slippage_k": 0.0005, "vol_penalty": 0.002,  "max_leverage": 3.0, "max_position": 50},
    "RTY": {"multiplier": 50,     "tick_size": 0.10,      "slippage_k": 0.001,  "vol_penalty": 0.004,  "max_leverage": 2.0, "max_position": 30},
    "CL":  {"multiplier": 1000,   "tick_size": 0.01,      "slippage_k": 0.002,  "vol_penalty": 0.005,  "max_leverage": 2.0, "max_position": 30},
    "NG":  {"multiplier": 10000,  "tick_size": 0.001,     "slippage_k": 0.003,  "vol_penalty": 0.008,  "max_leverage": 1.5, "max_position": 20},
    "GC":  {"multiplier": 100,    "tick_size": 0.10,      "slippage_k": 0.001,  "vol_penalty": 0.003,  "max_leverage": 2.5, "max_position": 40},
    "SI":  {"multiplier": 5000,   "tick_size": 0.005,     "slippage_k": 0.002,  "vol_penalty": 0.006,  "max_leverage": 2.0, "max_position": 25},
    "HG":  {"multiplier": 25000,  "tick_size": 0.0005,    "slippage_k": 0.0015, "vol_penalty": 0.004,  "max_leverage": 2.5, "max_position": 35},
    "ZN":  {"multiplier": 1000,   "tick_size": 0.015625,  "slippage_k": 0.0003, "vol_penalty": 0.0015, "max_leverage": 4.0, "max_position": 60},
    "ZB":  {"multiplier": 1000,   "tick_size": 0.03125,   "slippage_k": 0.0002, "vol_penalty": 0.001,  "max_leverage": 2.5, "max_position": 20},
    "ZT":  {"multiplier": 2000,   "tick_size": 0.0078125, "slippage_k": 0.0002, "vol_penalty": 0.001,  "max_leverage": 5.0, "max_position": 80},
}

HEADER = "# Inherits defaults from market_defaults.yaml — override-only fields listed.\n"

for ticker, s in MARKETS.items():
    doc = {
        "metadata": {
            "ticker": ticker,
            "contract_multiplier": s["multiplier"],
        },
        "contract_specs": {
            "tick_size": s["tick_size"],
            "tick_value": round(s["tick_size"] * s["multiplier"], 6),
        },
        "risk": {
            "slippage_k": s["slippage_k"],
            "vol_penalty": s["vol_penalty"],
            "max_leverage": s["max_leverage"],
            "max_position_size": s["max_position"],
        },
    }
    path = OUT / f"{ticker}.yaml"
    with open(path, "w") as f:
        f.write(HEADER + yaml.dump(doc, default_flow_style=False, sort_keys=False))
    print(f"  wrote {path.name} ({len(doc)} sections)")

print(f"\nDone — {len(MARKETS)} market files in {OUT}")