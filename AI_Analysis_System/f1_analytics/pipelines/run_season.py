from __future__ import annotations

from typing import Any

from sqlalchemy.engine import Engine

from ..models_finish_group import simulate_finish_group_probs
from ..models_pace_band import simulate_lap_delta_bands
from .pipeline_season import run_season_pipeline
from .run_meeting import (
    OpenF1Client,
    _build_fact_race_result,
    _build_feat_metrics,
    _compute_qpi,
    _save_outputs_to_db,
)


def run_season(engine: Engine, year: int) -> list[dict[str, Any]]:
    client = OpenF1Client()
    return run_season_pipeline(
        year=year,
        openf1_get=client.get,
        build_fact_race_result=_build_fact_race_result,
        build_feat_metrics=_build_feat_metrics,
        compute_qpi=_compute_qpi,
        simulate_lap_delta_bands=simulate_lap_delta_bands,
        simulate_finish_group_probs=simulate_finish_group_probs,
        save_outputs=lambda **kwargs: _save_outputs_to_db(engine=engine, **kwargs),
        min_driver_count=10,
    )

