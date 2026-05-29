import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    data: pl.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    _stage_log: list[dict[str, Any]] = field(default_factory=list)

    def log_stage(self, stage_name: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        rows = self.data.height if self.data is not None else 0
        cols = len(self.data.columns) if self.data is not None else 0
        entry = {
            "stage": stage_name,
            "timestamp": ts,
            "rows": rows,
            "columns": cols,
        }
        self._stage_log.append(entry)
        logger.info(
            "[PIPELINE] Stage complete: %s | rows=%d cols=%d",
            stage_name, rows, cols,
        )
