# 📊 Performance Benchmark Results

## Test Configuration

- **Hardware**: Apple Silicon (M4)
- **OS**: macOS
- **Model**: Whisper Large-v3 (3GB)
- **Audio Format**: M4A → WAV (16kHz mono)
- **Date**: 2025-07-01

## Benchmark Results

### Test File 1: Short Audio (1.2 minutes)
**File**: "Everyone is looking for the balance between freedom and structure.m4a"
- **Duration**: 73.6 seconds
- **File Size**: ~3MB

| Method | Time | Speed Factor | Notes |
|--------|------|--------------|-------|
| Core ML + Metal | 8.87s | 8.3x real-time | Neural Engine + GPU |
| Metal GPU Only | 9.50s | 7.7x real-time | GPU acceleration |
| CPU Only | ~12.7s | 5.8x real-time | Fallback mode |

### Test File 2: Long Audio (5.9 minutes) ⭐
**File**: "Miguel A1 drawing talk.m4a"
- **Duration**: 355.4 seconds (5.9 minutes)
- **File Size**: 9.7MB

| Method | Time | Speed Factor | Performance Gain |
|--------|------|--------------|------------------|
| 🥇 **Core ML + Metal** | **35.40s** | **10.0x real-time** | **+39.9% vs CPU** |
| 🥈 **Metal GPU Only** | **46.70s** | **7.6x real-time** | **+20.7% vs CPU** |
| 🥉 **CPU Only** | **58.86s** | **6.0x real-time** | *baseline* |

## Key Findings

### 🧠 Core ML Advantages
- **24.2% faster** than Metal GPU alone on longer audio
- **Neural Engine utilization** for encoder processing
- **Better sustained performance** on longer sequences
- **Automatic model detection** and loading

### ⚡ Metal GPU Performance
- Excellent **baseline performance** (7.6x real-time)
- **Good fallback** when Core ML unavailable
- **Consistent performance** across different audio lengths

### 💻 CPU Performance
- **Reliable fallback** option (6.0x real-time)
- Still **faster than real-time** processing
- Good for **compatibility** on non-Apple Silicon

## Performance Scaling

### Speed vs Audio Length
- **Short audio (1.2 min)**: Core ML shows minimal advantage
- **Long audio (5.9 min)**: Core ML shows significant 24% improvement
- **Conclusion**: Core ML benefits increase with audio length

### Real-Time Factors
- **Core ML**: 8.3x → 10.0x (scales better)
- **Metal**: 7.7x → 7.6x (consistent)
- **CPU**: 5.8x → 6.0x (stable)

## Technical Insights

### Core ML Model Detection
```
whisper_init_state: loading Core ML model from './whisper.cpp/models/ggml-large-v3-encoder.mlmodelc'
whisper_init_state: Core ML model loaded
system_info: COREML = 1 | Metal : EMBED_LIBRARY = 1
```

### Processing Breakdown (Long Audio)
| Component | Core ML | Metal Only | CPU Only |
|-----------|---------|------------|----------|
| Model Load | ~0.9s | ~0.8s | ~0.8s |
| Encoding | ~29s | ~40s | ~50s |
| Decoding | ~5s | ~6s | ~8s |
| **Total** | **35.4s** | **46.7s** | **58.9s** |

## Memory Usage

- **Model Size**: 3.09 GB (Large-v3)
- **GPU Memory**: ~3.1 GB allocated
- **Core ML Overhead**: Minimal additional memory
- **Peak RAM**: ~4-5 GB during processing

## Recommendations

### ✅ Optimal Setup
1. **Use Core ML + Metal** for best performance
2. **Ensure .mlmodelc model** is available
3. **Use longer audio files** to maximize Core ML benefits

### 🔄 Fallback Strategy
1. **Core ML + Metal** (preferred)
2. **Metal GPU only** (excellent fallback)
3. **CPU only** (compatibility mode)

## Hardware Specifications

- **Chip**: Apple M4
- **GPU**: Apple M4 (10-core GPU)
- **Neural Engine**: 16-core Neural Engine
- **Memory**: Unified memory architecture
- **Metal Family**: MTLGPUFamilyApple9

## Conclusion

The Core ML + Metal GPU combination delivers:
- ⚡ **10x real-time performance** on long audio
- 🧠 **24% improvement** over Metal GPU alone
- 🏆 **Best-in-class speed** for Apple Silicon
- 🔄 **Automatic optimization** with graceful fallbacks

This represents a **production-ready, professional-grade** transcription system optimized for Apple Silicon.
