#!/usr/bin/env python3

import numpy as np
import pyaudio
import sys
import time
import torch
from scipy.io import wavfile

# Import the ONNX inference helper (mirrors structure of inference.py)
from onnx_inference import predict_endpoint, RATE  # RATE reused for consistency

# --- Configuration ---
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
STOP_MS = 1000
PRE_SPEECH_MS = 200
MAX_DURATION_SECONDS = 16
VAD_THRESHOLD = 0.7
TEMP_OUTPUT_WAV = "temp_output.wav"

# ------------------------------------------------------------------
# Load Silero VAD once
# ------------------------------------------------------------------
try:
    print("Attempting to load Silero VAD model…")
    torch.hub.set_dir("./.torch_hub")
    vad_model, _ = torch.hub.load("snakers4/silero-vad", model="silero_vad", onnx=False, trust_repo=True)
    print("✅  Silero VAD model loaded.")
except Exception as e:
    print(f"❌  Error loading Silero VAD model: {e}")
    sys.exit(1)


def record_and_predict():
    """
    Manages real-time audio recording, voice activity detection,
    and calls the prediction function on detected speech segments.
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )
    print(f"\n🎤 Listening for speech... (VAD Threshold: {VAD_THRESHOLD})")

    audio_buffer = []
    silence_frames = 0
    speech_start_time = None
    speech_triggered = False

    try:
        while True:
            data = stream.read(CHUNK)
            audio_np = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_np.astype(np.float32) / 32767.0

            speech_prob = vad_model(torch.from_numpy(audio_float32), RATE).item()
            is_speech = speech_prob > VAD_THRESHOLD

            if is_speech:
                if not speech_triggered:
                    silence_frames = 0
                speech_triggered = True
                if speech_start_time is None:
                    speech_start_time = time.time()
                audio_buffer.append((time.time(), audio_float32))
            else:
                if speech_triggered:
                    audio_buffer.append((time.time(), audio_float32))
                    silence_frames += 1
                    if silence_frames * (CHUNK / RATE) >= STOP_MS / 1000:
                        speech_triggered = False
                        stream.stop_stream()
                        process_speech_segment(audio_buffer, speech_start_time)
                        audio_buffer = []
                        speech_start_time = None
                        print(f"\n🎤 Listening for speech... (VAD Threshold: {VAD_THRESHOLD})")
                        stream.start_stream()
                else:
                    audio_buffer.append((time.time(), audio_float32))
                    max_buffer_time = (PRE_SPEECH_MS + STOP_MS) / 1000 + 2.0
                    while audio_buffer and audio_buffer[0][0] < time.time() - max_buffer_time:
                        audio_buffer.pop(0)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def process_speech_segment(audio_buffer, speech_start_time):
    """
    Extracts the relevant audio segment from the buffer and sends it for prediction.
    """
    if not audio_buffer:
        return

    start_marker = speech_start_time - (PRE_SPEECH_MS / 1000)
    start_index = 0
    for i, (t, _) in enumerate(audio_buffer):
        if t >= start_marker:
            start_index = i
            break
    
    segment_audio_chunks = [chunk for _, chunk in audio_buffer[start_index:]]
    segment_audio = np.concatenate(segment_audio_chunks)

    if len(segment_audio) / RATE > MAX_DURATION_SECONDS:
        segment_audio = segment_audio[: int(MAX_DURATION_SECONDS * RATE)]

    if len(segment_audio) > 0:
        wavfile.write(TEMP_OUTPUT_WAV, RATE, (segment_audio * 32767).astype(np.int16))
        print(f"   Processing speech segment of {len(segment_audio) / RATE:.2f} seconds...")
        result = predict_endpoint(segment_audio)
        print("   ----------------------------------------")
        print(f"   Prediction: {'Complete Utterance' if result['prediction'] == 1 else 'Incomplete Utterance'}")
        print(f"   Probability of 'Complete': {result['probability']:.4f}")
        print(f"   Inference time: {result['inference_ms']:.1f} ms")
        print("   ----------------------------------------")
    else:
        print("   Captured empty audio segment, skipping prediction.")

if __name__ == "__main__":
    record_and_predict()
