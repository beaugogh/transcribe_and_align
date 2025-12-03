import math
import os

import webrtcvad
from pydub import AudioSegment

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes
CHANNELS = 1


def load_mp3_as_pcm(path: str) -> AudioSegment:
    """
    Load MP3 (or any audio) and convert it to:
    - 16 kHz
    - mono
    - 16-bit PCM
    """
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(SAMPLE_RATE)
    audio = audio.set_channels(CHANNELS)
    audio = audio.set_sample_width(SAMPLE_WIDTH)
    return audio


def detect_segments_ms(audio: AudioSegment,
                       silence_ms: int = 1000,
                       frame_ms: int = 30,
                       vad_mode: int = 3):
    """
    Use WebRTC VAD to find segment boundaries, return a list of
    (start_ms, end_ms) that together exactly cover the whole audio.
    """

    if frame_ms not in (10, 20, 30):
        raise ValueError("frame_ms must be 10, 20, or 30 ms for WebRTC VAD.")

    vad = webrtcvad.Vad(vad_mode)

    samples_per_frame = int(SAMPLE_RATE * frame_ms / 1000)
    bytes_per_frame = samples_per_frame * SAMPLE_WIDTH * CHANNELS

    raw = audio.raw_data
    total_ms = len(audio)  # pydub gives duration in ms

    # Number of *full* frames we can read
    n_frames = len(raw) // bytes_per_frame

    frames_needed = math.ceil(silence_ms / frame_ms)

    segment_start_frame = 0
    silent_run = 0
    segment_starts = [0]  # in frames

    for i in range(n_frames):
        offset = i * bytes_per_frame
        frame = raw[offset: offset + bytes_per_frame]

        if len(frame) != bytes_per_frame:
            break

        is_speech = vad.is_speech(frame, SAMPLE_RATE)

        if not is_speech:
            silent_run += 1
        else:
            silent_run = 0

        # If we’ve seen enough consecutive non-speech, split AFTER this frame
        if silent_run >= frames_needed:
            cut_frame = i + 1  # cut after current frame
            segment_starts.append(cut_frame)
            silent_run = 0

    # Convert frame-based boundaries to ms, ensuring full coverage
    segments_ms = []
    for idx, start_frame in enumerate(segment_starts):
        start_ms = start_frame * frame_ms
        if idx + 1 < len(segment_starts):
            end_frame = segment_starts[idx + 1]
            end_ms = end_frame * frame_ms
        else:
            # last segment goes to end of audio (including any leftover <frame_ms)
            end_ms = total_ms
        # Clamp to [0, total_ms]
        start_ms = max(0, min(start_ms, total_ms))
        end_ms = max(0, min(end_ms, total_ms))
        if end_ms > start_ms:
            segments_ms.append([start_ms, end_ms])

    return segments_ms


def merge_short_segments(segments_ms, min_chunk_sec=5.0):
    """
    Merge segments shorter than min_chunk_sec with neighbors
    WITHOUT dropping any time.

    segments_ms: list of [start_ms, end_ms], contiguous & covering the full audio.
    """
    min_ms = int(min_chunk_sec * 1000)
    segs = [list(s) for s in segments_ms]

    i = 0
    while i < len(segs):
        s, e = segs[i]
        dur = e - s

        # If segment long enough or only one segment, keep as is
        if dur >= min_ms or len(segs) == 1:
            i += 1
            continue

        # Too short -> merge
        if i == 0:
            # Merge into next: [s0,e0] + [s1,e1] -> [s0,e1], drop seg0
            segs[1][0] = s
            del segs[0]
            # i stays 0 to re-check new seg[0]
        else:
            # Merge into previous: [s_{i-1}, e_{i-1}] + [s_i, e_i]
            segs[i-1][1] = e
            del segs[i]
            i -= 1  # re-check previous after enlargement

    return segs


def split_audio_on_silence(path: str,
                         silence_ms: int = 1000,
                         frame_ms: int = 30,
                         vad_mode: int = 3,
                         min_chunk_sec: float = 5.0):
    """
    High-level function:
    1. Detect segments via VAD
    2. Merge segments shorter than min_chunk_sec
    3. Export each as a WAV file

    No samples are dropped: final chunks cover exactly [0, total_audio_duration].
    """
    audio = load_mp3_as_pcm(path)

    # 1. Detect raw segments from VAD
    segments_ms = detect_segments_ms(audio,
                                     silence_ms=silence_ms,
                                     frame_ms=frame_ms,
                                     vad_mode=vad_mode)

    # 2. Merge too-short segments
    merged_segments = merge_short_segments(segments_ms,
                                           min_chunk_sec=min_chunk_sec)

    # 3. Export each segment using time slicing (no raw-byte concat)
    base = os.path.splitext(path)[0]
    base = f"{base}_chunks"
    os.makedirs(base, exist_ok=True)
    for idx, (start_ms, end_ms) in enumerate(merged_segments, start=1):
        chunk = audio[start_ms:end_ms]
        out_path = os.path.join(base, f"chunk_{idx}.wav")
        chunk.export(out_path, format="wav")
        print(f"Saved: {out_path}  [{start_ms} ms → {end_ms} ms]")

    print("Done. Total chunks:", len(merged_segments))
    return base



if __name__ == "__main__":
    split_audio_on_silence(
        "/home/bo/workspace/whisper/tasks/sample.mp3",
        silence_ms=1000,   # split at pauses > 1s
        frame_ms=30,       # 30 ms VAD frames
        vad_mode=3,        # most sensitive
        min_chunk_sec=5.0  # merge segments < 5 seconds
    )
