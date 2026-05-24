#!/usr/bin/env python3
import re
from pathlib import Path

PROJECT_ROOT = Path.cwd()
RUN_PY = PROJECT_ROOT / "run.py"
CONFIG_PY = PROJECT_ROOT / "config.py"

# 1. Update run.py
if RUN_PY.exists():
    with open(RUN_PY, "r", encoding="utf-8") as f:
        content = f.read()
    new_content = re.sub(r'-m src\.cli', '-m quant.cli', content)
    new_content = re.sub(r'-m src\.analytics', '-m quant.analytics', new_content)
    if new_content != content:
        with open(RUN_PY, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("✅ Updated run.py: src.cli → quant.cli")
    else:
        print("run.py already uses quant.cli")

# 2. Add missing config attribute
if CONFIG_PY.exists():
    with open(CONFIG_PY, "r", encoding="utf-8") as f:
        content = f.read()
    if "ROLL_WINDOW_MIN_ROWS" not in content:
        # Insert after existing ROLL_WINDOWS_DAILY or at end
        insert = "\nconfig.ROLL_WINDOW_MIN_ROWS = 20\n"
        if "ROLL_WINDOWS_DAILY" in content:
            content = content.replace("ROLL_WINDOWS_DAILY", "ROLL_WINDOWS_DAILY" + insert)
        else:
            content += insert
        with open(CONFIG_PY, "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ Added config.ROLL_WINDOW_MIN_ROWS = 20")
    else:
        print("config.py already has ROLL_WINDOW_MIN_ROWS")

# 3. (Optional) Remove .bak files if any
for bak in PROJECT_ROOT.rglob("*.bak"):
    bak.unlink()
    print(f"Removed {bak}")

print("\nNow run: python run.py")