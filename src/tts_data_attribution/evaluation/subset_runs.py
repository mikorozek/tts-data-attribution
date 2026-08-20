from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ..experiment import Plan, TrainingRunManifest
from ..models import is_lora_checkpoint_complete


@dataclass(frozen=True)
class CompletedSubsetRun:
    subset_id: str
    training_run_name: str
    manifest: TrainingRunManifest
    target: Path


def discover_completed_subset_runs(
    experiment_directory: str | Path,
    plan: Plan,
    reference_training_run: TrainingRunManifest,
) -> list[CompletedSubsetRun]:
    experiment_directory = Path(experiment_directory)
    completed: dict[str, CompletedSubsetRun] = {}
    for run_directory in sorted((experiment_directory / "training-runs").iterdir()):
        target = run_directory / "target"
        if not is_lora_checkpoint_complete(target):
            continue
        training_run = TrainingRunManifest.from_yaml(run_directory / "manifest.yaml")
        subset_id = training_run.training_set
        if subset_id not in plan.subsets:
            continue
        if subset_id in completed:
            other_run = completed[subset_id].training_run_name
            raise ValueError(
                f"multiple completed runs for {subset_id}: "
                f"{other_run}, {run_directory.name}"
            )
        expected = replace(reference_training_run, training_set=subset_id)
        if training_run != expected:
            raise ValueError(
                f"training configuration for {run_directory.name} "
                "differs from the projection training run"
            )
        completed[subset_id] = CompletedSubsetRun(
            subset_id=subset_id,
            training_run_name=run_directory.name,
            manifest=training_run,
            target=target,
        )
    return [
        completed[subset_id] for subset_id in plan.subsets if subset_id in completed
    ]
