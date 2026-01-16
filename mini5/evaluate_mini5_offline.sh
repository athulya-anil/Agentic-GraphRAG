#!/bin/bash
# Run evaluation in offline mode (no HuggingFace network calls)

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "Running evaluation in OFFLINE mode..."
python3 evaluate_mini5.py "$@"
