import argparse

import librosa
import numpy as np

from smart_turn.inference import DEFAULT_ONNX_MODEL_PATH, predict_endpoint


def main():
    parser = argparse.ArgumentParser(description="Run Smart Turn endpoint prediction on an audio file.")
    parser.add_argument("file_path", help="Path to an audio file")
    parser.add_argument(
        "--model",
        default=DEFAULT_ONNX_MODEL_PATH,
        help="Path to the Smart Turn ONNX model",
    )
    args = parser.parse_args()

    try:
        print(f"Loading audio file: {args.file_path}")
        # Load the audio file with original sample rate
        audio, sr = librosa.load(args.file_path, sr=None, mono=True)

        print(f"Loaded audio with sample rate: {sr} Hz, duration: {len(audio) / sr:.2f} seconds")

        # If needed, resample to 16kHz (the model's expected sample rate)
        if sr != 16000:
            print(f"Resampling from {sr}Hz to 16000Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Convert audio to float32 if not already
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Make sure the audio is in the expected range [-1, 1]
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))

        # Call the prediction function with both audio and sample rate
        print("Running endpoint prediction...")
        result = predict_endpoint(audio, onnx_path=args.model)

        # Display results
        print("\nResults:")
        print(f"Prediction: {'Complete' if result['prediction'] == 1 else 'Incomplete'}")
        print(f"Probability of complete: {result['probability']:.4f}")

    except Exception as e:
        print(f"Error processing audio file: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
