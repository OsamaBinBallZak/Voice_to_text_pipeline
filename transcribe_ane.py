#!/usr/bin/env python3
"""
High-performance transcription script optimized for Apple Neural Engine
Uses whisper.cpp with Core ML acceleration
"""
import os
import sys
import subprocess
import numpy as np
from pydub import AudioSegment
import noisereduce as nr
import tempfile

def transcribe_with_ane(audio_file, model_path=None, use_coreml=True):
    """
    Transcribe audio using ANE-optimized whisper.cpp
    """
    print("🚀 Starting ANE-optimized transcription...")
    
    # Default model path
    if not model_path:
        model_path = "./whisper.cpp/models/ggml-large-v3.bin"
    
    # Load and preprocess audio
    print("📂 Loading audio file...")
    audio = AudioSegment.from_file(audio_file)
    
    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Ensure 16kHz sample rate (Whisper requirement)
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    
    # Convert to numpy array for noise reduction
    samples = audio.get_array_of_samples()
    audio_np = np.array(samples).astype(np.float32)
    
    # Normalize
    audio_np = audio_np / np.max(np.abs(audio_np))
    
    print("🔧 Applying noise reduction...")
    # Apply noise reduction
    reduced_audio = nr.reduce_noise(y=audio_np, sr=16000)
    
    # Convert back to AudioSegment
    reduced_audio_int16 = (reduced_audio * 32767).astype(np.int16)
    processed_audio = AudioSegment(
        reduced_audio_int16.tobytes(),
        frame_rate=16000,
        sample_width=2,
        channels=1
    )
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_audio_path = temp_file.name
        processed_audio.export(temp_audio_path, format="wav")
    
    try:
        # Prepare whisper.cpp command
        whisper_cmd = [
            "./whisper.cpp/build/bin/whisper-cli",
            "-m", model_path,
            "-f", temp_audio_path,
            "--language", "auto",
            "--output-txt",
"--output-file", "transcription_output.txt",
            "--threads", str(os.cpu_count()),
        ]
        
        # Add acceleration options
        if not use_coreml:
            # Disable GPU to use CPU only
            whisper_cmd.append("--no-gpu")
            print("💻 Using CPU acceleration")
        else:
            print("⚡ Using Metal GPU acceleration")
        
        print("🎯 Running transcription...")
        result = subprocess.run(whisper_cmd, capture_output=True, text=True, cwd="/Users/tiurihartog/Downloads/audio-transcription-pipeline")
        
        if result.returncode == 0:
            print("✅ Transcription completed!")
            print("\n📝 Transcription:")
            print("=" * 50)
            print(result.stdout)
            
            # Try to read the output file if it was created
            try:
                with open("/Users/tiurihartog/Downloads/audio-transcription-pipeline/transcription_output.txt", "r") as f:
                    transcription_text = f.read()
                    if transcription_text.strip():
                        print("\n📄 Cleaned transcription:")
                        print("=" * 50)
                        print(transcription_text)
            except FileNotFoundError:
                pass
            
        else:
            print("❌ Transcription failed!")
            print("Error:", result.stderr)
            return None
        
        
    finally:
        # Clean up temporary file
        os.unlink(temp_audio_path)
        # Clean up output files
        for ext in [".txt", ".srt", ".vtt"]:
            try:
                os.unlink(f"/Users/tiurihartog/Downloads/audio-transcription-pipeline/transcription_output{ext}")
            except FileNotFoundError:
                pass

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_ane.py <audio_file> [--no-coreml]")
        print("Example: python transcribe_ane.py recording.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    use_coreml = "--no-coreml" not in sys.argv
    
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file '{audio_file}' not found!")
        sys.exit(1)
    
    transcribe_with_ane(audio_file, use_coreml=use_coreml)

if __name__ == "__main__":
    main()
