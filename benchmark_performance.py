#!/usr/bin/env python3
"""
Comprehensive performance benchmark for Core ML vs Metal GPU transcription
"""
import tempfile
import time
import shutil
import os
import subprocess
from pydub import AudioSegment

def run_benchmark(audio_file):
    """Run comprehensive benchmark comparing Core ML vs Metal-only performance"""
    
    print("🚀 Starting Comprehensive Performance Benchmark")
    print("=" * 60)
    
    # Check audio file
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    # Get audio info
    audio = AudioSegment.from_file(audio_file)
    duration_seconds = len(audio) / 1000
    duration_minutes = duration_seconds / 60
    
    print(f"📊 Audio File Info:")
    print(f"   File: {os.path.basename(audio_file)}")
    print(f"   Duration: {duration_minutes:.1f} minutes ({duration_seconds:.1f} seconds)")
    print(f"   Sample rate: {audio.frame_rate} Hz")
    print(f"   Channels: {audio.channels}")
    print(f"   File size: {os.path.getsize(audio_file) / (1024*1024):.1f} MB")
    print()
    
    # Prepare audio
    print("📂 Preprocessing audio...")
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        audio.export(temp_file.name, format='wav')
        temp_audio_path = temp_file.name
        print(f"   Preprocessed audio saved to: {temp_audio_path}")
    
    # Paths
    coreml_path = './whisper.cpp/models/ggml-large-v3-encoder.mlmodelc'
    backup_path = './whisper.cpp/models/ggml-large-v3-encoder.mlmodelc.backup'
    model_path = './whisper.cpp/models/ggml-large-v3.bin'
    
    results = {}
    
    try:
        # Test 1: Core ML + Metal GPU
        print("\n🧠 Test 1: Core ML + Metal GPU")
        print("-" * 40)
        if os.path.exists(coreml_path):
            start_time = time.time()
            result = subprocess.run([
                './whisper.cpp/build/bin/whisper-cli',
                '-m', model_path,
                '-f', temp_audio_path,
                '--no-timestamps'
            ], capture_output=True, text=True)
            end_time = time.time()
            
            coreml_time = end_time - start_time
            results['coreml'] = {
                'time': coreml_time,
                'transcription': result.stdout.strip()[:100] + "...",
                'stderr': result.stderr
            }
            
            print(f"   ✅ Time: {coreml_time:.2f} seconds")
            print(f"   ⚡ Speed: {duration_seconds/coreml_time:.1f}x real-time")
            print(f"   📝 Preview: {result.stdout.strip()[:80]}...")
        else:
            print("   ❌ Core ML model not found")
        
        # Test 2: Metal GPU only (temporarily move Core ML model)
        print("\n⚡ Test 2: Metal GPU Only")
        print("-" * 40)
        
        # Backup Core ML model temporarily
        if os.path.exists(coreml_path):
            shutil.move(coreml_path, backup_path)
        
        start_time = time.time()
        result = subprocess.run([
            './whisper.cpp/build/bin/whisper-cli',
            '-m', model_path,
            '-f', temp_audio_path,
            '--no-timestamps'
        ], capture_output=True, text=True)
        end_time = time.time()
        
        metal_time = end_time - start_time
        results['metal'] = {
            'time': metal_time,
            'transcription': result.stdout.strip()[:100] + "...",
            'stderr': result.stderr
        }
        
        print(f"   ✅ Time: {metal_time:.2f} seconds")
        print(f"   ⚡ Speed: {duration_seconds/metal_time:.1f}x real-time")
        print(f"   📝 Preview: {result.stdout.strip()[:80]}...")
        
        # Restore Core ML model
        if os.path.exists(backup_path):
            shutil.move(backup_path, coreml_path)
        
        # Test 3: CPU Only
        print("\n💻 Test 3: CPU Only")
        print("-" * 40)
        
        start_time = time.time()
        result = subprocess.run([
            './whisper.cpp/build/bin/whisper-cli',
            '-m', model_path,
            '-f', temp_audio_path,
            '--no-gpu',
            '--no-timestamps'
        ], capture_output=True, text=True)
        end_time = time.time()
        
        cpu_time = end_time - start_time
        results['cpu'] = {
            'time': cpu_time,
            'transcription': result.stdout.strip()[:100] + "...",
            'stderr': result.stderr
        }
        
        print(f"   ✅ Time: {cpu_time:.2f} seconds")
        print(f"   ⚡ Speed: {duration_seconds/cpu_time:.1f}x real-time")
        print(f"   📝 Preview: {result.stdout.strip()[:80]}...")
        
    finally:
        # Cleanup
        os.unlink(temp_audio_path)
        # Ensure Core ML model is restored
        if os.path.exists(backup_path):
            shutil.move(backup_path, coreml_path)
    
    # Performance Summary
    print("\n🏆 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Audio Duration: {duration_minutes:.1f} minutes ({duration_seconds:.1f} seconds)")
    print()
    
    if 'coreml' in results and 'metal' in results and 'cpu' in results:
        coreml_time = results['coreml']['time']
        metal_time = results['metal']['time']
        cpu_time = results['cpu']['time']
        
        print(f"🧠 Core ML + Metal:  {coreml_time:8.2f}s  ({duration_seconds/coreml_time:5.1f}x real-time)")
        print(f"⚡ Metal GPU Only:   {metal_time:8.2f}s  ({duration_seconds/metal_time:5.1f}x real-time)")
        print(f"💻 CPU Only:        {cpu_time:8.2f}s  ({duration_seconds/cpu_time:5.1f}x real-time)")
        print()
        
        # Performance gains
        coreml_vs_metal = ((metal_time - coreml_time) / metal_time) * 100
        metal_vs_cpu = ((cpu_time - metal_time) / cpu_time) * 100
        coreml_vs_cpu = ((cpu_time - coreml_time) / cpu_time) * 100
        
        print("📈 Performance Gains:")
        print(f"   Core ML vs Metal GPU: {coreml_vs_metal:+.1f}%")
        print(f"   Metal GPU vs CPU:    {metal_vs_cpu:+.1f}%")  
        print(f"   Core ML vs CPU:      {coreml_vs_cpu:+.1f}%")
        print()
        
        # Speed rankings
        times = [
            ("🥇 Core ML + Metal", coreml_time),
            ("🥈 Metal GPU Only", metal_time),
            ("🥉 CPU Only", cpu_time)
        ]
        times.sort(key=lambda x: x[1])
        
        print("🏅 Speed Rankings:")
        for i, (name, time_val) in enumerate(times, 1):
            print(f"   {i}. {name}: {time_val:.2f}s")
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python benchmark_performance.py <audio_file>")
        print("Example: python benchmark_performance.py '/path/to/audio.m4a'")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    run_benchmark(audio_file)
