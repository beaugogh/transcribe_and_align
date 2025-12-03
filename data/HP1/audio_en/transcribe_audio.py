import os
import json
import whisper


def transcribe_chunks(directory=".", whisper_model="large"):
    """
    Transcribe all *chunk_*.wav files in the directory using Whisper large model,
    and save JSON files next to them.
    """

    # Load Whisper large model (this can take ~2GB VRAM)
    print("Loading Whisper large model...")
    model = whisper.load_model(whisper_model)

    # Find all chunk wav files
    chunk_files = [
        f for f in os.listdir(directory)
        if f.endswith(".wav") and "chunk" in f
    ]

    if not chunk_files:
        print("No chunk wav files found.")
        return

    print(f"Found {len(chunk_files)} chunks.")

    for wav in sorted(chunk_files):
        wav_path = os.path.join(directory, wav)
        json_path = wav_path.replace(".wav", ".json")

        print(f"\nTranscribing {wav} ...")

        # Run Whisper transcription
        result = model.transcribe(
            wav_path,
            fp16=False,            # CPU users need this; GPU users can remove it
            word_timestamps=True   # include detailed timestamps
        )

        # Prepare JSON structure
        output = {
            "chunk_file": wav,
            "text": result.get("text", ""),
            "language": result.get("language", ""),
            "segments": [
                {
                    "id": seg.get("id"),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text")
                }
                for seg in result.get("segments", [])
            ]
        }

        # Save JSON next to the chunk file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Saved transcript → {json_path}")

    print("\nDone! All chunks transcribed.")


if __name__ == "__main__":
    transcribe_chunks("/home/bo/workspace/whisper/tasks/sample_chunks")