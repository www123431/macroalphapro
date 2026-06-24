"""Sector pipeline agent subpackage.

Migrated from engine/sector_pipeline.run_sector_pipeline on 2026-05-03 as P0
step 2 of the agent-infra adoption sweep
(memory/project_agentic_orchestration_v1.md).

`prepare_sector_inputs` (pure ETL) stays in engine/sector_pipeline as a helper
imported by this agent.
"""
from engine.agents.sector_pipeline.agent import SectorPipelineAgent

__all__ = ["SectorPipelineAgent"]
