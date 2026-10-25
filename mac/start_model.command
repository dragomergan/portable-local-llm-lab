#!/bin/bash
# Start KoboldCPP (macOS, Apple Silicon) with a model that you choose.
#
# Put your .gguf model files in the "models" folder, next to the folder that holds
# this script. The script lists the chat models it finds and asks which one to run.
# A file whose name contains "embed" is used as the embedding model (needed for RAG).
# Files whose name contains "mmproj" (image add-ons) are ignored.
#
# To skip the menu:  ./start_model.command <model-file-name.gguf>

# Settings you may want to change
PORT=5001
THREADS=8
CONTEXT=8192
KOBOLD=./koboldcpp-mac-arm64

cd "$(dirname "$0")" || exit 1
MODELS=../models

if [ ! -f "$KOBOLD" ]; then
  echo "KoboldCPP program not found: $KOBOLD"
  echo "Download it from https://github.com/LostRuins/koboldcpp/releases and put it in this folder."
  exit 1
fi
if [ ! -x "$KOBOLD" ]; then
  chmod +x "$KOBOLD" 2>/dev/null || { echo "Cannot make $KOBOLD executable. See the troubleshooting section of the README."; exit 1; }
fi

# Find the models. Embedding models and "mmproj" files are not offered as chat models.
models=()
embed=""
for file in "$MODELS"/*.gguf; do
  [ -f "$file" ] || continue
  lower=$(basename "$file" | tr '[:upper:]' '[:lower:]')
  case "$lower" in
    *embed*)  [ -z "$embed" ] && embed="$file" ;;
    *mmproj*) ;;
    *)        models+=("$file") ;;
  esac
done

selected=""
if [ -n "$1" ]; then
  if [ -f "$MODELS/$1" ]; then
    selected="$MODELS/$1"
  else
    echo "Model not found: $MODELS/$1"
    exit 1
  fi
elif [ ${#models[@]} -eq 0 ]; then
  echo "No chat model (.gguf) found in $MODELS"
  echo "Download one (see the README), put it in that folder, and run this script again."
  exit 1
elif [ ${#models[@]} -eq 1 ]; then
  selected="${models[0]}"
  echo "Only one model found: $(basename "$selected")"
else
  echo
  echo "Available models:"
  i=1
  for m in "${models[@]}"; do
    echo "  $i) $(basename "$m")"
    i=$((i + 1))
  done
  echo
  while true; do
    read -r -p "Type the number of the model and press Enter: " choice
    if [[ "$choice" =~ ^[0-9]+$ ]]; then
      choice=$((10#$choice))
      if [ "$choice" -ge 1 ] && [ "$choice" -le ${#models[@]} ]; then
        selected="${models[$((choice - 1))]}"
        break
      fi
    fi
    echo "Please type a number between 1 and ${#models[@]}."
  done
fi

echo
echo "Starting with model: $(basename "$selected")"
args=(--model "$selected" --port "$PORT" --threads "$THREADS" --contextsize "$CONTEXT")
if [ -n "$embed" ]; then
  echo "Embedding model:     $(basename "$embed")"
  args+=(--embeddingsmodel "$embed")
else
  echo "No embedding model found, so RAG features will not work (put a file with \"embed\" in its name into $MODELS)."
fi
echo

"$KOBOLD" "${args[@]}"

echo
read -r -p "KoboldCPP has stopped. Press Enter to close this window. " _
