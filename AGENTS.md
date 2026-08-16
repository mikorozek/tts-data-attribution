# AGENTS.md

## Mission

Build a reusable framework for running and validating data-attribution experiments. The framework must support repeated experiments with different PyTorch models, datasets, training procedures, parameter subsets, attribution settings, and evaluation responses.

Reusable code must not hard-code a particular model architecture, dataset schema, adaptation method, loss, query type, or experiment size. Concrete behavior belongs in focused implementation modules, scripts, and consumed configurations.

## Core design principles

- **Model-agnostic:** gradient collection must accept a user-supplied `torch.nn.Module` together with an explicit differentiable per-example objective.
- **Parameter-agnostic:** users must be able to select the exact named parameters or modules whose gradients are collected. The selector may target adapters, full-model weights, or another subset without changing attribution code.
- **Dataset-agnostic:** raw datasets are converted into a documented canonical interface. Dataset-specific parsing, filtering, grouping, and collation stay in focused dataset modules.
- **Objective-agnostic:** the experiment provides the per-example training loss and held-out response. The framework must not assume cross-entropy, teacher forcing, classification logits, or waveform generation.
- **Configuration-driven:** choices that vary between runs are expressed in validated configuration, resolved against the loaded objects, and saved with each run.
- **Reproducible:** every artifact is tied to immutable example IDs, source revisions, configuration, random seeds, masks, and code/environment versions.
- **Composable:** training, gradient collection, projection, attribution, and evaluation are separate components with explicit inputs and outputs.

## Required framework capabilities

### Data integration

A dataset integration must be able to:

- convert raw records into stable example IDs, model inputs, targets, and metadata;
- expose grouping keys needed to prevent leakage, such as conversation, speaker, document, or source IDs;
- validate and materialize configurable train, development, reference, and query partitions;
- provide task-specific collation without imposing that collation on other datasets;
- persist resolved manifests and content hashes independently of raw data storage.

The framework should make it straightforward to add a new dataset integration that produces the canonical interface without modifying attribution or LDS code.

### Model and objective integration

A model integration must provide:

- a factory that constructs a fresh `torch.nn.Module` at a reproducible initial state;
- task-specific batch-to-model-input preparation;
- an explicit per-example differentiable objective;
- a held-out response used by evaluation;
- optional training and serialization hooks when the model cannot use the generic runner directly.

The gradient collector operates on the returned `nn.Module`, objective tensor, and selected named parameters. It must not depend on a specific Hugging Face class or internal model path.

### Parameter selection and gradient collection

Parameter selection must be configurable through explicit include/exclude selectors and module/parameter predicates. Before collection, the framework must resolve selectors to exact parameter names and shapes, reject empty or ambiguous selections, and save the resolved selection manifest.

Gradient collection must support at least a reliable batch-size-one path and should allow optimized batched backends when their correctness is tested. It must preserve example identity through projection and storage.

### Training and counterfactual runs

The implemented training commands must support:

- one or more target-model training runs;
- independently initialized and trained subset/counterfactual runs;
- configurable subset masks and seeds;
- identical initial states where required by the experiment;
- clean optimizer reset and deterministic artifact naming;
- restartable jobs and cluster/job-array execution.

Never warm-start a subset model from a model that has already seen examples excluded from that subset unless the experiment explicitly studies that different intervention.

### Attribution

TrackStar is the first attribution backend. Its implementation should consume generic per-example gradient features and selected optimizer statistics, rather than model-specific code. Projection, optimizer correction, covariance correction, normalization, checkpoint aggregation, and storage must be independently configurable and testable.

Additional attribution backends should be addable behind the same user-facing interface.

### Evaluation

LDS is the first counterfactual validation backend. It should accept:

- an attribution vector for each query;
- subset membership masks;
- observed scalar outcomes from corresponding subset models;
- configurable correlation, aggregation, and uncertainty estimation.

LDS code must not assume what the scalar outcome means. An integration may use negative loss, margin, accuracy, perceptual score, or another documented response.

## Configuration and resolved manifests

Each implemented component owns its configuration schema and documentation. `AGENTS.md` must not contain run-specific values, model parameter names, dataset split sizes, or hyperparameters.

At runtime, configuration must be resolved to concrete objects and saved. Depending on the component, the resolved manifest includes:

- dataset record IDs, groups, partitions, and hashes;
- model/source revisions and initial-state hash;
- exact selected parameter names, shapes, dtypes, and counts;
- objective and response definitions;
- optimizer, schedule, masks, checkpoints, and seeds;
- projection/correction settings;
- software and hardware environment.

A new experimental variant should normally require a new consumed configuration or focused implementation module, not a modification to unrelated reusable code.

## Correctness requirements

- The objective used for training and gradient attribution must be explicit and documented.
- Never replace a training objective with sampled generation or an arbitrary differentiable surrogate unless that is the declared experiment.
- Preserve per-example identity and reduction semantics from data loading through gradients, attribution, and evaluation.
- Prevent partition leakage using integration-provided grouping metadata.
- Verify parameter selection, gradient finiteness, projection determinism, serialization, mask construction, and evaluation aggregation with tests.
- Run a small end-to-end pilot before scaling counterfactual training.
- Clearly distinguish data used in the controlled experiment from any unknown upstream pretraining data.
- Do not present a budget-reduced validation protocol as a reproduction of a larger published checkpoint budget.

## Separation of repository roles

```text
src/tts_data_attribution/  importable implementation, including the tda CLI
references/                provenance, papers, checksums, and licenses
third_party/               pinned vendored upstream source
tests/                     executable behavior checks
data/ and artifacts/       ignored source dataset and model assets
experiments/               ignored local experiment workspaces
```

Add a module, command, or configuration only with its first concrete consumer. The `tda` CLI package only parses arguments and composes reusable implementation; dataset, model, and training logic must stay importable and testable without it. An experiment is defined once by `tda experiment init`; its manifest, plan, speaker embeddings, and run outputs live together in its directory under `experiments/`. Experiment workspaces are local and never tracked; each run stays reproducible through the manifests saved with it.

## Repository map

```text
AGENTS.md                  framework-wide design and working rules
README.md                  human-facing framework overview
docs/specs/                reviewed framework interface specifications
src/tts_data_attribution/  reusable framework and optional integrations
references/                core papers, source manifests, and licenses
data/                      ignored source dataset assets and derived dataset products
artifacts/                 ignored downloaded upstream model assets
experiments/               ignored local experiment workspaces: config, plan, data, runs
third_party/               pinned vendored upstream source and documented project patches
tests/                     unit, integration, and reproducibility tests
```

Model links, dataset links, local asset paths, and immutable source details belong in `references/sources.yaml`. Focused protocols and known issues belong in the relevant implementation or documentation.

## Repository rules

- Keep source, configuration keys, tests, and technical documentation in English.
- Formatting and linting are defined once in `pyproject.toml` under `[tool.ruff]` and obeyed by every editor and agent. Before any commit, run `uvx ruff format --check src tests` and `uvx ruff check src tests`; treat every finding as an error and fix it, never silence it.
- Do not add comments or docstrings to project-owned code or tests.
- Make code self-documenting through precise names, explicit types, and small focused units.
- Put construction and serialization behavior on the domain type it belongs to; reserve module-level functions for behavior that has no natural owner.
- Keep large data, weights, optimizer states, checkpoints, gradient features, generated media, and run logs out of Git.
- Never silently update pinned upstream code or artifacts.
- Model behavior that the framework needs — training forward passes, per-example objectives, gradient collection, adapter injection, speaker conditioning, serialization — is added directly to the vendored upstream model classes in `third_party/`, not to wrapper classes in `src/`. Keep each edit minimal, preserve licenses and upstream provenance, record every change in that vendor directory's `PATCHES.md`, and cover changed behavior with tests. Never bump the pinned upstream without carrying the recorded changes over.
- Every completed run must be reproducible from its resolved manifest and immutable inputs.
- Add reusable behavior only when it is generic; otherwise keep it in a focused dataset, model, script, or configuration component.

## Core references

- TrackStar paper: <https://arxiv.org/abs/2410.17413>; untracked local working copy: `references/papers/trackstar-2410.17413v3.pdf`.
- TrackStar code: <https://github.com/pair-code/pretraining-tda>.
- TRAK / LDS paper: <https://arxiv.org/abs/2303.14186>; untracked local working copy: `references/papers/trak-2303.14186v2.pdf`.
- TRAK code: <https://github.com/MadryLab/trak>.

The URLs, checksums, and byte sizes in `references/sources.yaml` are the tracked provenance for these untracked local PDFs.
