#!/usr/bin/env bash
# Register the local GGUF with Ollama as 'openhealth-doctor'.
# Run once after cloning or updating the GGUF artifact.
#
# Usage:
#   cd models/
#   bash setup_model.sh
#
# After this, set LLM_MODEL=openhealth-doctor in your .env file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="openhealth-doctor"
GGUF_FILE="gguf_doctor_model_llama-unsloth.Q4_K_M.gguf"
MODELFILE="Modelfile"

cd "$SCRIPT_DIR"

# Verify ollama is available
if ! command -v ollama &>/dev/null; then
  echo "ERROR: ollama not found. Install from https://ollama.com/download" >&2
  exit 1
fi

# Verify GGUF exists
if [[ ! -f "$GGUF_FILE" ]]; then
  echo "ERROR: $GGUF_FILE not found in $SCRIPT_DIR"
  echo "Download it from: https://huggingface.co/kevinjoythomas/medical-loratuned-chatbot-GGUF" >&2
  exit 1
fi

echo "Registering $MODEL_NAME with Ollama..."
ollama create "$MODEL_NAME" -f "$MODELFILE"

echo ""
echo "Done. Model '$MODEL_NAME' is ready."
echo ""
echo "Next step — add to your .env:"
echo "  LLM_MODEL=$MODEL_NAME"
