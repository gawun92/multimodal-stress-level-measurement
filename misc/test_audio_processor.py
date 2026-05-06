import numpy as np
from feature_extraction.audio_processor import load_audio, compute_mel_spectrogram, pad_or_truncate, normalize

def test_audio_pipeline():
    # Mock a 2-second 16kHz audio array (32000 samples)
    dummy_audio = np.random.randn(32000).astype(np.float32)

    # Test mel computation
    mel = compute_mel_spectrogram(dummy_audio)
    
    # 32000 samples / 512 hop length = 62.5 -> ~63 frames
    assert mel.shape[0] == 128, f"Expected 128 mel bins, got {mel.shape[0]}"
    assert 60 <= mel.shape[1] <= 65, f"Expected ~63 frames, got {mel.shape[1]}"
    
    # Test padding
    padded_mel = pad_or_truncate(mel, max_frames=1876)
    assert padded_mel.shape == (128, 1876), f"Expected (128, 1876) after padding, got {padded_mel.shape}"
    
    # Test truncation
    long_audio = np.random.randn(16000 * 70).astype(np.float32) # 70 seconds
    long_mel = compute_mel_spectrogram(long_audio)
    trunc_mel = pad_or_truncate(long_mel, max_frames=1876)
    assert trunc_mel.shape == (128, 1876), f"Expected (128, 1876) after truncation, got {trunc_mel.shape}"
    
    # Test normalization
    norm_mel = normalize(padded_mel)
    assert np.isclose(norm_mel.mean(), 0.0, atol=1e-5), f"Mean should be 0, got {norm_mel.mean()}"
    assert np.isclose(norm_mel.std(), 1.0, atol=1e-2), f"Std should be 1, got {norm_mel.std()}"
    
    print("All audio processor tests passed!")

if __name__ == "__main__":
    test_audio_pipeline()
