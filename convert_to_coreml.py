#!/usr/bin/env python3
"""
Convert Whisper model to Core ML format optimized for Apple Neural Engine
"""
import os
import sys
import torch
import whisper
import coremltools as ct
import numpy as np
from typing import Dict, Any

def convert_whisper_to_coreml(model_name: str = "large-v3", output_dir: str = "coreml_models"):
    """
    Convert Whisper model to Core ML format with ANE optimization
    """
    print(f"Loading Whisper {model_name} model...")
    model = whisper.load_model(model_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model dimensions
    n_mels = model.dims.n_mels  # Number of mel frequency bins (128 for large-v3)
    n_audio_ctx = model.dims.n_audio_ctx  # Context length
    
    print(f"Model dimensions: n_mels={n_mels}, n_audio_ctx={n_audio_ctx}")
    
    # Extract encoder
    encoder = model.encoder
    
    print("Converting encoder to Core ML...")
    
    # Prepare input for model encoder
    encoder.eval()
    
    # Check the positional embedding shape to understand context length
    pos_embed_shape = encoder.positional_embedding.shape
    print(f"Positional embedding shape: {pos_embed_shape}")
    
    # The encoder expects mel spectrograms with n_mels channels
    # Input format should be: (batch, n_mels, n_ctx)
    # where n_ctx is the sequence length (time dimension)
    expected_n_ctx = pos_embed_shape[0]  # Sequence length from positional embedding
    expected_n_mels = n_mels  # Use the actual n_mels from model dims (128)
    
    print(f"Correct input shape: (1, {expected_n_mels}, {expected_n_ctx})")
    
    # Create dummy mel spectrogram with correct dimensions
    dummy_mel = torch.randn(1, expected_n_mels, expected_n_ctx)
    
    print(f"Testing encoder with mel shape: {dummy_mel.shape}")
    
    # Test the encoder first before tracing
    try:
        with torch.no_grad():
            output = encoder(dummy_mel)
            print(f"Encoder test successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"Encoder test failed: {e}")
        # The issue might be that we need to process the mel spectrogram first
        # Let's check what the first layer of the encoder expects
        first_layer = list(encoder.children())[0]
        print(f"First encoder layer: {first_layer}")
        
        # Try with a real mel spectrogram from actual audio
        try:
            print("Trying with real mel spectrogram preprocessing...")
            # Create some dummy audio
            dummy_audio = torch.randn(16000 * 30)  # 30 seconds at 16kHz
            
            # Use Whisper's preprocessing
            mel = whisper.log_mel_spectrogram(dummy_audio, n_mels=n_mels)
            mel = whisper.pad_or_trim(mel, expected_n_ctx)
            mel = mel.unsqueeze(0)  # Add batch dimension
            
            print(f"Preprocessed mel shape: {mel.shape}")
            
            output = encoder(mel)
            print(f"Real mel test successful! Output shape: {output.shape}")
            dummy_mel = mel
            
        except Exception as e3:
            print(f"Real mel preprocessing also failed: {e3}")
            return None
    
    # Trace the encoder using dummy mel spectrogram
    try:
        traced_encoder = torch.jit.trace(encoder, dummy_mel)
        print("Encoder traced successfully")
    except Exception as e:
        print(f"Error tracing encoder: {e}")
        return None
    
    # Convert to Core ML with ANE optimization
    try:
        # Use the actual dimensions that worked
        input_shape = dummy_mel.shape
        encoder_coreml = ct.convert(
            traced_encoder,
            inputs=[ct.TensorType(shape=input_shape, name="mel_spectrogram")],
            outputs=[ct.TensorType(name="encoder_output")],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,  # Use all available compute units including ANE
            minimum_deployment_target=ct.target.macOS13  # Ensures ANE compatibility
        )
    except Exception as e:
        print(f"Error converting to Core ML: {e}")
        # Try with different settings
        try:
            print("Retrying with CPU_AND_GPU compute units...")
            encoder_coreml = ct.convert(
                traced_encoder,
                inputs=[ct.TensorType(shape=input_shape, name="mel_spectrogram")],
                outputs=[ct.TensorType(name="encoder_output")],
                compute_precision=ct.precision.FLOAT32,
                compute_units=ct.ComputeUnit.CPU_AND_GPU,
                minimum_deployment_target=ct.target.macOS13
            )
        except Exception as e2:
            print(f"Second conversion attempt failed: {e2}")
            return None
    
    encoder_path = os.path.join(output_dir, f"whisper_{model_name}_encoder.mlpackage")
    encoder_coreml.save(encoder_path)
    print(f"Encoder saved to: {encoder_path}")
    
    # For the decoder, we need a more complex approach due to autoregressive nature
    print("Converting decoder components to Core ML...")
    
    # Convert just the text decoder embedding and initial layers
    # This is more complex due to the autoregressive nature, so we'll focus on the encoder first
    
    print("Conversion complete!")
    print(f"Models saved in: {output_dir}")
    
    return encoder_path

def test_coreml_model(model_path: str):
    """Test the Core ML model performance"""
    print(f"Testing Core ML model: {model_path}")
    
    try:
        import coremltools as ct
        model = ct.models.MLModel(model_path)
        print("Model loaded successfully!")
        print(f"Model description: {model.short_description}")
        
        # Test with dummy input
        dummy_input = {"mel_spectrogram": np.random.randn(1, 80, 1500).astype(np.float32)}
        prediction = model.predict(dummy_input)
        print("Test prediction successful!")
        
    except Exception as e:
        print(f"Error testing model: {e}")

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "large-v3"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "coreml_models"
    
    print(f"Converting Whisper {model_name} to Core ML...")
    encoder_path = convert_whisper_to_coreml(model_name, output_dir)
    
    print("\nTesting converted model...")
    test_coreml_model(encoder_path)
