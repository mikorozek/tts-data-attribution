.PHONY: papers reference-experiment-references reference-experiment-assets status

papers:
	python scripts/download_papers.py

reference-experiment-references:
	python experiments/qwen3_tts_dailytalk/scripts/download_references.py

reference-experiment-assets:
	bash experiments/qwen3_tts_dailytalk/scripts/download_assets.sh

status:
	git status --short --branch
