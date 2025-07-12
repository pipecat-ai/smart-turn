import numpy as np
import pyaudio
import sys
from scipy.io import wavfile
import time
import torch
from silero_vad import load_silero_vad

# Import the new inference script
from coreml_inference import predict_endpoint

# --- Configuration ---
RATE = 16000
CHUNK = 512  # Adjusted CHUNK size to match Silero VAD expectation for 16kHz
FORMAT = pyaudio.paInt16
CHANNELS = 1
STOP_MS = 1000
PRE_SPEECH_MS = 200
MAX_DURATION_SECONDS = 16  # Maximum duration (seconds) compatible with Core ML export
VAD_THRESHOLD = 0.7  # Threshold for speech detection
TEMP_OUTPUT_WAV = "temp_output.wav"

# --- Load Silero VAD Model ---
try:
    print("Attempting to load Silero VAD model...")
    torch.hub.set_dir("./.torch_hub")  # Set a specific hub directory for debugging
    MODEL = load_silero_vad()
    print("Silero VAD model loaded successfully.")
except Exception as e:
    print(f"Error loading Silero VAD model: {e}")
    print(
        "Please ensure you have an internet connection and that torch and torchaudio are installed correctly."
    )
    print("You can try installing them using: pip install torch torchaudio")
    print("If the issue persists, try clearing the torch hub cache:")
    print("import torch; print(torch.hub.get_dir())  # Find the cache directory")
    print("Then manually delete the 'snakers4_silero-vad_...' folder.")
    sys.exit(1)


def record_and_predict():
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print("Clearing audio buffer and listening for speech...")

    audio_buffer = []
    silence_frames = 0
    speech_start_time = None
    speech_triggered = False

    try:
        while True:
            data = stream.read(CHUNK)
            audio_np = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_np.astype(np.float32) / np.iinfo(np.int16).max

            # Use Silero VAD model directly to get probability
            speech_prob = MODEL(torch.from_numpy(audio_float32).unsqueeze(0), RATE).item()
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
                        speech_stop_time = time.time()

                        # Stop the stream before processing
                        stream.stop_stream()

                        process_speech_segment(audio_buffer, speech_start_time, speech_stop_time)
                        audio_buffer = []
                        speech_start_time = None

                        # Restart the stream after processing
                        stream.start_stream()
                        print("Listening for speech...")
                else:
                    # Keep buffering some silence before potential speech starts
                    audio_buffer.append((time.time(), audio_float32))
                    # Keep the buffer size reasonable, assuming CHUNK is small
                    max_buffer_time = (
                        PRE_SPEECH_MS + STOP_MS
                    ) / 1000 + MAX_DURATION_SECONDS  # Some extra buffer
                    while audio_buffer and audio_buffer[0][0] < time.time() - max_buffer_time:
                        audio_buffer.pop(0)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def process_speech_segment(audio_buffer, speech_start_time, speech_stop_time):
    if not audio_buffer:
        return

    # Find start and end indices for the segment
    start_time = speech_start_time - (PRE_SPEECH_MS / 1000)
    start_index = 0
    for i, (t, _) in enumerate(audio_buffer):
        if t >= start_time:
            start_index = i
            break

    end_index = len(audio_buffer) - 1

    # Extract the audio segment
    segment_audio_chunks = [chunk for _, chunk in audio_buffer[start_index : end_index + 1]]
    segment_audio = np.concatenate(segment_audio_chunks)

    # Remove (STOP_MS - 200)ms from the end of the segment
    samples_to_remove = int((STOP_MS - 200) / 1000 * RATE)
    segment_audio = segment_audio[:-samples_to_remove]

    # Limit maximum duration
    if len(segment_audio) / RATE > MAX_DURATION_SECONDS:
        segment_audio = segment_audio[: int(MAX_DURATION_SECONDS * RATE)]

    # No resampling needed as both recording and prediction use 16000 Hz
    segment_audio_resampled = segment_audio

    if len(segment_audio_resampled) > 0:
        # Save the audio for debugging purposes
        wavfile.write(TEMP_OUTPUT_WAV, RATE, (segment_audio_resampled * 32767).astype(np.int16))
        print(f"Processing speech segment of length {len(segment_audio) / RATE:.2f} seconds...")

        # Call the new predict_endpoint function with the audio data
        start_time = time.perf_counter()
        result = predict_endpoint(segment_audio_resampled)
        end_time = time.perf_counter()
        print(result)

        print("--------")
        print(f"Prediction: {'Complete' if result['prediction'] == 1 else 'Incomplete'}")
        print(f"Probability of complete: {result['probability']:.4f}")
        print(f"Prediction took {(end_time - start_time) * 1000:.2f}ms seconds")
    else:
        print("Captured empty audio segment, skipping prediction.")


if __name__ == "__main__":
    record_and_predict()
