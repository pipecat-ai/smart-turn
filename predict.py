import librosa
import sys
import numpy as np

from inference import predict_endpoint


def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        print("Usage: python predict.py <path_to_file>")
        sys.exit(1)

    try:
        print(f"Loading audio file: {file_path}")
        # Load the audio file with original sample rate
        audio, sr = librosa.load(file_path, sr=None, mono=True)

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
        result = predict_endpoint(audio)

        # Display results
        print("\nResults:")
        print(f"Prediction: {'Complete' if result['prediction'] == 1 else 'Incomplete'}")
        print(f"Probability of complete: {result['probability']:.4f}")

    except Exception as e:
        print(f"Error processing audio file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()