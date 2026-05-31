from stable_asr.data.audio import inspect_wav, load_wav_mono, synth_tone, write_wav_mono


def test_wav_roundtrip(tmp_path) -> None:
    path = tmp_path / "tone.wav"
    samples = synth_tone(0.05, sample_rate=16000, frequency=220.0, seed=1)

    write_wav_mono(path, samples, sample_rate=16000)
    loaded, sample_rate = load_wav_mono(path)

    assert sample_rate == 16000
    assert len(loaded) == len(samples)
    assert max(abs(sample) for sample in loaded) > 0.0

    info = inspect_wav(path)
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert info.duration_sec > 0
