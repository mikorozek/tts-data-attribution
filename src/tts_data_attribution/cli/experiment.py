from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..dataset import DATASETS, UtteranceDataset
from ..experiment import ExperimentManifest, Plan
from ..models import EXPERIMENT_ENCODERS
from .errors import CommandError


def register(subparsers: argparse._SubParsersAction) -> None:
    experiment_parser = subparsers.add_parser("experiment", help="experiment workspace")
    experiment_subparsers = experiment_parser.add_subparsers(required=True)

    init_parser = experiment_subparsers.add_parser(
        "init", help="sample an experiment from a dataset"
    )
    init_parser.add_argument("name")
    init_parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    init_parser.add_argument("--data-root", type=Path, required=True)
    init_parser.add_argument("--training-pool-size", type=int, required=True)
    init_parser.add_argument("--validation-pool-size", type=int, required=True)
    init_parser.add_argument("--query-pool-size", type=int, required=True)
    init_parser.add_argument("--subset-count", type=int, required=True)
    init_parser.add_argument("--subset-size", type=int, required=True)
    init_parser.add_argument("--speaker-count", type=int, required=True)
    init_parser.add_argument("--seed", type=int, required=True)
    init_parser.add_argument("--root", type=Path, default=Path("experiments"))
    init_parser.set_defaults(run=run_init)

    encode_parser = experiment_subparsers.add_parser(
        "encode", help="encode the utterances sampled for an experiment"
    )
    encode_parser.add_argument("name")
    encode_parser.add_argument(
        "--model", choices=sorted(EXPERIMENT_ENCODERS), required=True
    )
    encode_parser.add_argument("--model-path", type=Path, required=True)
    encode_parser.add_argument("--device", default="cuda:0")
    encode_parser.add_argument("--batch-size", type=int, default=16)
    encode_parser.add_argument("--root", type=Path, default=Path("experiments"))
    encode_parser.set_defaults(run=run_encode)


def run_init(arguments: argparse.Namespace) -> None:
    directory = arguments.root / arguments.name
    if directory.exists():
        raise CommandError(f"experiment {arguments.name} already exists at {directory}")

    manifest = ExperimentManifest(
        dataset=arguments.dataset,
        data_root=arguments.data_root,
        training_pool_size=arguments.training_pool_size,
        validation_pool_size=arguments.validation_pool_size,
        query_pool_size=arguments.query_pool_size,
        subset_count=arguments.subset_count,
        subset_size=arguments.subset_size,
        speaker_count=arguments.speaker_count,
        seed=arguments.seed,
    )
    dataset = load_dataset(manifest)

    try:
        plan = Plan.sample(manifest, dataset.get_records())
    except ValueError as error:
        raise CommandError(str(error)) from error

    directory.mkdir(parents=True)
    manifest.to_yaml(directory / "manifest.yaml")
    plan.to_json(directory / "plan.json")
    print(f"experiment {arguments.name} created at {directory}")


def run_encode(arguments: argparse.Namespace) -> None:
    directory = arguments.root / arguments.name
    manifest_path = directory / "manifest.yaml"
    plan_path = directory / "plan.json"
    if not manifest_path.is_file() or not plan_path.is_file():
        raise CommandError(
            f"experiment {arguments.name} is incomplete; run experiment init first"
        )
    try:
        manifest = ExperimentManifest.from_yaml(manifest_path)
        plan = Plan.from_json(plan_path)
    except (
        KeyError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        raise CommandError(
            f"cannot read experiment {arguments.name}: {error}"
        ) from error
    dataset = load_dataset(manifest)
    selected_ids = (
        plan.training_pool
        + plan.validation_pool
        + plan.query_pool
        + list(plan.references.values())
    )
    try:
        dataset.get_records_by_ids(selected_ids)
    except KeyError as error:
        raise CommandError(
            f"sampled utterance missing from dataset: {error}"
        ) from error

    encoding_path = directory / "encoding.yaml"
    encoding = {
        "model": arguments.model,
        "model_path": arguments.model_path.as_posix(),
    }
    if encoding_path.is_file():
        try:
            existing_encoding = yaml.safe_load(
                encoding_path.read_text(encoding="utf-8")
            )
        except yaml.YAMLError as error:
            raise CommandError(f"cannot read {encoding_path}: {error}") from error
        if existing_encoding != encoding:
            raise CommandError(
                f"experiment {arguments.name} was already encoded with "
                f"{existing_encoding}"
            )

    output = directory / "sampled_utterances_encoded.jsonl"
    try:
        encoded_ids = (
            UtteranceDataset.from_jsonl(output).ids() if output.is_file() else set()
        )
    except (TypeError, json.JSONDecodeError) as error:
        raise CommandError(f"cannot read {output}: {error}") from error
    pending_ids = [
        identifier for identifier in selected_ids if identifier not in encoded_ids
    ]

    speaker_path = directory / "speaker_embeddings.pt"
    speaker_embeddings = (
        torch.load(speaker_path, map_location="cpu", weights_only=True)
        if speaker_path.is_file()
        else {}
    )
    speakers_are_encoded = all(
        speaker in speaker_embeddings for speaker in plan.references
    )

    if not pending_ids and speakers_are_encoded:
        print(f"all {len(selected_ids)} sampled utterances are already encoded")
        return

    try:
        encoder_type = EXPERIMENT_ENCODERS[arguments.model]()
        encoder = encoder_type.from_pretrained(arguments.model_path, arguments.device)
    except (OSError, TypeError, ValueError) as error:
        raise CommandError(
            f"cannot load {arguments.model} from {arguments.model_path}: {error}"
        ) from error

    encoding_path.write_text(yaml.safe_dump(encoding, sort_keys=True), encoding="utf-8")

    completed = 0
    for batch in DataLoader(
        dataset,
        sampler=pending_ids,
        batch_size=arguments.batch_size,
        collate_fn=list,
    ):
        encoded_batch = encoder.encode_utterances(batch, manifest.data_root)
        UtteranceDataset(encoded_batch).to_jsonl(output, append=True)
        completed += len(batch)
        print(f"encoded {completed}/{len(pending_ids)} utterances", flush=True)

    for speaker, reference_id in plan.references.items():
        if speaker in speaker_embeddings:
            continue
        reference = dataset.get_records_by_ids([reference_id])[0]
        speaker_embeddings[speaker] = encoder.encode_speaker(
            manifest.data_root / reference.audio_path
        )
    torch.save(speaker_embeddings, speaker_path)
    print(f"encoded experiment ready at {directory}")


def load_dataset(manifest: ExperimentManifest):
    try:
        return DATASETS[manifest.dataset](manifest.data_root)
    except (OSError, TypeError, ValueError) as error:
        raise CommandError(
            f"cannot load {manifest.dataset} from {manifest.data_root}: {error}"
        ) from error
