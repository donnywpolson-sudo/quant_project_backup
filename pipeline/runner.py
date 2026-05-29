import logging
from collections.abc import Callable
from typing import Any

from pipeline.tracking.state import PipelineState

logger = logging.getLogger(__name__)


Stage = Callable[[PipelineState], PipelineState]


class PipelineRunner:
    def __init__(self, stages: list[Stage]) -> None:
        self.stages = stages
        self._executed: list[str] = []

    def run(self, state: PipelineState) -> PipelineState:
        if not self.stages:
            logger.warning("PipelineRunner.run called with zero stages.")
            return state

        for stage in self.stages:
            stage_name = getattr(stage, "__name__", stage.__class__.__name__)
            logger.info("[PIPELINE] Executing stage: %s", stage_name)
            state = stage(state)
            state.log_stage(stage_name)
            self._executed.append(stage_name)

        return state

    @property
    def executed_stages(self) -> list[str]:
        return list(self._executed)
