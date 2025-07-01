# ⚡ High-Performance Audio Transcription Pipeline

A lightning-fast audio transcription system optimized for Apple Silicon, achieving **10x real-time transcription** using Core ML + Metal GPU acceleration.

## 🚀 Features

- **Ultra-fast processing**: 10x real-time transcription speed
- **Core ML + Metal acceleration**: Leverages Apple's Neural Engine and GPU
- **High accuracy**: Uses Whisper Large-v3 model for superior transcription quality
- **Noise reduction**: Built-in audio preprocessing for cleaner results
- **Multi-format support**: Works with MP3, M4A, WAV, FLAC, OGG
- **Clean output**: Timestamped transcription with cleaned text versions
- **Simple interface**: Single command transcription

## 📊 Performance Benchmarks

### **Comprehensive Benchmark Results** (5.9-minute audio file)

| Method | Time | Speed | Performance Gain |
|--------|------|-------|------------------|
| 🥇 **Core ML + Metal** | **35.4s** | **10.0x real-time** | **+39.9% vs CPU** |
| 🥈 **Metal GPU Only** | **46.7s** | **7.6x real-time** | **+20.7% vs CPU** |
| 🥉 **CPU Only** | **58.9s** | **6.0x real-time** | *baseline* |

### **Key Performance Insights**
- **Core ML optimization**: 24.2% faster than Metal GPU alone
- **Neural Engine advantage**: Dedicated ML silicon for encoder processing
- **Sustained performance**: Maintains speed even on longer audio files
- **Model**: Whisper Large-v3 (3GB) with Core ML acceleration

## 🛠 Setup

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3)
- Python 3.8+
- Homebrew

### Installation
```bash
# Install dependencies
brew install cmake

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Build whisper.cpp (if not already built)
cd whisper.cpp
mkdir build && cd build
cmake .. -DGGML_METAL=ON
make -j

# Download model (if not present)
cd ../models
# Large-v3 model should be at: ggml-large-v3.bin
```

## 🎯 Usage

### Basic Transcription
```bash
# Activate environment
source venv/bin/activate

# Transcribe with Metal GPU acceleration (default)
python transcribe_ane.py "path/to/your/audio.m4a"

# Force CPU-only processing
python transcribe_ane.py "path/to/your/audio.m4a" --no-coreml
```

### Example Output
```
🚀 Starting ANE-optimized transcription...
📂 Loading audio file...
🔧 Applying noise reduction...
⚡ Using Metal GPU acceleration
🎯 Running transcription...
✅ Transcription completed!

📝 Transcription:
==================================================

[00:00:00.000 --> 00:00:12.180]   There is a balance between freedom and structure...
```

## 📁 Project Structure

```
audio-transcription-pipeline/
├── transcribe_ane.py          # Main transcription script
├── requirements.txt           # Python dependencies
├── whisper.cpp/              # Whisper.cpp submodule
│   ├── build/                # Compiled binaries
│   └── models/               # Model files
├── venv/                     # Python virtual environment
└── README.md                 # This file
```

## 🔧 Technical Details

### Core Components
- **whisper.cpp**: C++ implementation of OpenAI Whisper
- **Metal GPU**: Apple's GPU compute framework
- **Noise Reduction**: noisereduce library for audio preprocessing
- **Audio Processing**: pydub for format conversion and normalization

### Processing Pipeline
1. **Audio Loading**: Load and convert to mono 16kHz WAV
2. **Noise Reduction**: Apply spectral gating noise reduction
3. **Normalization**: Normalize audio levels
4. **Transcription**: Metal GPU-accelerated Whisper inference
5. **Output**: Timestamped and cleaned transcription

### Model Information
- **Model**: Whisper Large-v3
- **Size**: ~3GB
- **Languages**: Multilingual with auto-detection
- **Quality**: State-of-the-art transcription accuracy

## 🎉 Achievements

This pipeline demonstrates the incredible power of Apple Silicon for AI workloads:

- ✅ **10x real-time transcription** with Core ML + Metal GPU
- ✅ **35 seconds to process 6 minutes** of audio
- ✅ **24% faster than Metal GPU alone** with Core ML optimization
- ✅ **Production-ready accuracy** with Whisper Large-v3
- ✅ **Clean, simple interface** for easy usage
- ✅ **Robust preprocessing** for various audio qualities
- ✅ **Neural Engine utilization** for maximum performance

## 🚫 What We Removed

- ❌ **Speaker diarization**: Removed messy, inaccurate speaker attribution
- ❌ **Complex dependencies**: Simplified to core functionality
- ❌ **Slow processing**: Eliminated CPU-bound operations where possible

## 📝 Notes

- Metal GPU acceleration provides the best performance on Apple Silicon
- The pipeline automatically handles audio format conversion
- Noise reduction improves transcription quality for low-quality audio
- Output includes both timestamped and cleaned text versions

---

**Built for speed, accuracy, and simplicity on Apple Silicon** 🍎⚡
