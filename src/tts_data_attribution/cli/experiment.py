from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from ..dataset import UtteranceDataset
from ..experiment import ExperimentManifest, Plan
from .errors import CommandError

MODELS = ["qwen3-tts"]


def register(subparsers: argparse._SubParsersAction) -> None:
    experiment_parser = subparsers.add_parser("experiment", help="experiment workspace")
    experiment_subparsers = experiment_parser.add_subparsers(required=True)
    init_parser = experiment_subparsers.add_parser(
        "init", help="define an experiment: dataset, model, sampled subsets, references"
    )
    init_parser.add_argument("name")
    init_parser.add_argument("--dataset", type=Path, required=True)
    init_parser.add_argument("--audio-root", type=Path, required=True)
    init_parser.add_argument("--model", choices=MODELS, required=True)
    init_parser.add_argument("--model-path", type=Path, required=True)
    init_parser.add_argument("--training-pool-size", type=int, required=True)
    init_parser.add_argument("--subset-count", type=int, required=True)
    init_parser.add_argument("--subset-size", type=int, required=True)
    init_parser.add_argument("--speaker-count", type=int, required=True)
    init_parser.add_argument("--seed", type=int, required=True)
    init_parser.add_argument("--device", default="cuda:0")
    init_parser.add_argument("--root", type=Path, default=Path("experiments"))
    init_parser.set_defaults(run=run_init)


def run_init(arguments: argparse.Namespace) -> None:
    directory = arguments.root / arguments.name
    if directory.exists():
        raise CommandError(f"experiment {arguments.name} already exists at {directory}")
    manifest = ExperimentManifest(
        dataset=arguments.dataset,
        audio_root=arguments.audio_root,
        model=arguments.model,
        model_path=arguments.model_path,
        training_pool_size=arguments.training_pool_size,
        subset_count=arguments.subset_count,
        subset_size=arguments.subset_size,
        speaker_count=arguments.speaker_count,
        seed=arguments.seed,
    )

    if not manifest.dataset.is_file():
        raise CommandError(
            f"encoded dataset not found at {manifest.dataset}; run: tda data encode ..."
        )
    try:
        dataset = UtteranceDataset.from_jsonl(manifest.dataset)
    except (TypeError, json.JSONDecodeError) as error:
        raise CommandError(
            f"{manifest.dataset} is not an encoded utterance file: {error}"
        ) from error

    try:
        plan = Plan.sample(manifest, dataset)
    except ValueError as error:
        raise CommandError(str(error)) from error

    if importlib.util.find_spec("qwen_tts") is None:
        raise CommandError(
            "loading the model needs the vendored qwen-tts package; "
            "run: uv run --group qwen tda experiment init ..."
        )
    if not manifest.model_path.is_dir():
        raise CommandError(f"model directory not found at {manifest.model_path}")
    import librosa
    from qwen_tts import Qwen3TTSModel

    try:
        model = Qwen3TTSModel.from_pretrained(
            str(manifest.model_path), device_map=arguments.device
        )
    except (OSError, TypeError, ValueError) as error:
        raise CommandError(
            f"cannot load {manifest.model} from {manifest.model_path}: {error}"
        ) from error

    utterances = {utterance.id: utterance for utterance in dataset}
    sample_rate = model.model.speaker_encoder_sample_rate
    speaker_embeddings: dict[str, torch.Tensor] = {}
    for speaker, reference_id in plan.references.items():
        wav_path = manifest.audio_root / utterances[reference_id].audio_path
        wav, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)
        speaker_embeddings[speaker] = model.model.extract_speaker_embedding(
            wav, sample_rate
        ).cpu()

    directory.mkdir(parents=True)
    manifest.to_yaml(directory / "manifest.yaml")
    plan.to_json(directory / "plan.json")
    torch.save(speaker_embeddings, directory / "speaker_embeddings.pt")
    print(f"experiment {arguments.name} created at {directory}")
