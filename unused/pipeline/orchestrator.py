"""
pipeline/orchestrator.py — Full pipeline orchestration.

Chains all 12 stages in causal order via PipelineRunner.
Stage N only imports from Stage N-1 or core/ infrastructure.
"""

import logging
from typing import Any

from pipeline.tracking.state import PipelineState
from pipeline.runner import PipelineRunner

logger = logging.getLogger(__name__)


def stage_ingest(state: PipelineState) -> PipelineState:
    """
    Stage 1-2: Raw data ingestion + continuous contract adjustment.
    Calls pipeline/ingest/stage.py which internally applies
    build_continuous_series before resampling (Patch 2).
    """
    from pipeline.ingest.stage import ingestion_stage

    return ingestion_stage(state)


def stage_normalize(state: PipelineState) -> PipelineState:
    """
    Stage 3-4: Session normalization + HTF alignment + gap filter.
    Session filter, resample to 5m/1h/1d from adjusted data,
    align HTF streams, apply corrected gap filter (Patch 1).
    """
    from pipeline.session.normalization import normalization_stage

    return normalization_stage(state)


def stage_features(state: PipelineState) -> PipelineState:
    """
    Stage 5-6: Feature engineering + target generation.
    Available when used outside walkforward (e.g., single-pass runs).
    """
    from pipeline.features.engine import generate_features

    df = generate_features(state.data)
    state.data = df
    state.log_stage("features")
    return state


def stage_regime(state: PipelineState) -> PipelineState:
    """
    Stage 7: HMM regime detection (optional — only in run_hmm mode).
    Requires 1H resampled data in state metadata.
    """
    logger.info("[ORCHESTRATOR] Regime stage: reserved for walkforward-integrated HMM")
    return state


def stage_meta(state: PipelineState) -> PipelineState:
    """
    Stage 8: Meta-labeling target generation.
    Applied inside walkforward, exposed here for standalone use.
    """
    from pipeline.meta.meta_label import add_meta_label_target

    if "target_meta" not in state.data.columns:
        state.data = add_meta_label_target(state.data)
    state.log_stage("meta_label")
    return state


def stage_execution(state: PipelineState) -> PipelineState:
    """
    Stage 9: Execution simulation.
    Applied inside walkforward, exposed here for standalone use.
    """
    from pipeline.execution.simulator import simulate_execution_classification

    if "pnl" not in state.data.columns and "position" in state.data.columns:
        state.data = simulate_execution_classification(state.data)
    state.log_stage("execution")
    return state


def stage_walkforward(state: PipelineState) -> PipelineState:
    """
    Stage 10: Walkforward training + backtest + PnL simulation.
    Orchestrates features, targets, regime, meta-label, execution
    across sequential walkforward folds.
    """
    from pipeline.walkforward.walkforward import run_walkforward

    data_glob = state.metadata.get("data_glob", "")
    manifest_path = state.metadata.get("manifest_path", None)
    output_dir = state.metadata.get("output_dir", "results")
    folds = state.metadata.get("folds", 6)

    run_walkforward(
        data_glob=data_glob,
        cache_path=state.metadata.get("cache_path", None),
        manifest_path=manifest_path,
        output_dir=output_dir,
        folds=folds,
    )
    state.log_stage("walkforward")
    return state


def stage_analytics(state: PipelineState) -> PipelineState:
    """
    Stage 11: Aggregate analytics and metrics.
    """
    from pipeline.analytics.aggregate import run_aggregation

    output_dir = state.metadata.get("output_dir", "results")
    run_aggregation(output_dir)
    state.log_stage("analytics")
    return state


def stage_track(state: PipelineState) -> PipelineState:
    """
    Stage 12: Experiment tracking — logs final state.
    """
    logger.info(
        "[ORCHESTRATOR] Pipeline complete: %d stages executed",
        len(state._stage_log),
    )
    for entry in state._stage_log:
        logger.info("  %s | rows=%d cols=%d", entry["stage"], entry["rows"], entry["columns"])
    return state


def get_default_stages() -> list:
    """Return the standard causal-order stage list."""
    return [
        stage_ingest,
        stage_normalize,
        stage_walkforward,
        stage_analytics,
        stage_track,
    ]


def run_pipeline(
    data_glob: str,
    cache_path: str = None,
    manifest_path: str = None,
    output_dir: str = "results",
    folds: int = 6,
    cross_asset_symbols: list = None,
    extra_metadata: dict = None,
) -> PipelineState:
    """
    Execute the full pipeline end-to-end.

    Args:
        data_glob: Glob pattern for raw parquet files.
        cache_path: Optional path for cached aligned data.
        manifest_path: Optional feature manifest path.
        output_dir: Directory for results output.
        folds: Number of walkforward folds.
        cross_asset_symbols: Secondary symbols for cross-asset features.
        extra_metadata: Additional metadata to store in PipelineState.

    Returns:
        PipelineState with full stage log and final data.
    """
    metadata: dict[str, Any] = {
        "data_glob": data_glob,
        "cache_path": cache_path,
        "manifest_path": manifest_path,
        "output_dir": output_dir,
        "folds": folds,
        "cross_asset_symbols": cross_asset_symbols or [],
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    import polars as pl

    state = PipelineState(data=pl.DataFrame(), metadata=metadata)
    stages = get_default_stages()
    runner = PipelineRunner(stages)
    logger.info("[ORCHESTRATOR] Starting pipeline: %s", data_glob)
    state = runner.run(state)
    logger.info("[ORCHESTRATOR] Pipeline finished successfully.")
    return state
