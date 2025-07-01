#!/bin/bash

# Audio Transcription Pipeline Launcher
# Usage: ./run_transcription.sh <audio_file> [--diarize]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Set model path
MODEL_PATH="$SCRIPT_DIR/whisper.cpp/models/ggml-large-v3.bin"

# Check if audio file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <audio_file> [--diarize]"
    echo "Example: $0 /path/to/your/audio.wav --diarize"
    exit 1
fi

AUDIO_FILE="$1"
DIARIZE_FLAG=""

# Check for diarize flag
if [ "$2" = "--diarize" ] || [ "$3" = "--diarize" ]; then
    DIARIZE_FLAG="--diarize"
fi

echo "Starting transcription..."
echo "Audio file: $AUDIO_FILE"
echo "Model: $MODEL_PATH"
echo "Diarization: ${DIARIZE_FLAG:-disabled}"
echo ""

# Run transcription
python transcribe.py "$AUDIO_FILE" "$MODEL_PATH" $DIARIZE_FLAG

echo ""
echo "Transcription complete!"
