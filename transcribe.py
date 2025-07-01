import os
import sys
import subprocess
import numpy as np
from pydub import AudioSegment
import noisereduce as nr
import whisper
from pyannote.audio import Pipeline

def main(audio_file, model_path=None, diarize=False):
    # Load audio
    audio = AudioSegment.from_file(audio_file)
    samples = audio.get_array_of_samples()
    sample_rate = audio.frame_rate

    # Noise reduction
    reduced_noise = nr.reduce_noise(y=np.array(samples).astype(np.float32), sr=sample_rate)

    # Save temporary reduced noise audio
    reduced_audio_file = "temp_reduced.wav"
    reduced_audio = AudioSegment(
        reduced_noise.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    reduced_audio.export(reduced_audio_file, format="wav")

    # Transcription
    if model_path and model_path.endswith('.bin'):
        # Use whisper.cpp via subprocess for .bin models
        whisper_cpp_path = "./whisper.cpp/bin/whisper-cli"
        cmd = [whisper_cpp_path, "-m", model_path, "-f", reduced_audio_file, "--language", "auto"]
        result_output = subprocess.run(cmd, capture_output=True, text=True)
        print("Transcription:")
        print(result_output.stdout)
    else:
        # Use Python whisper for other models
        model = whisper.load_model(model_path or "large-v3")
        result = model.transcribe(reduced_audio_file)
        print("Transcription:")
        print(result["text"])

    # Diarization
    if diarize:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        diarization = pipeline(reduced_audio_file)
        print("Speaker Diarization:")
        print(diarization)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python transcribe.py <audio_file> <model_path> [--diarize]")
        sys.exit(1)

    audio_file = sys.argv[1]
    model_path = sys.argv[2]
    diarize = "--diarize" in sys.argv

    main(audio_file, model_path, diarize)

