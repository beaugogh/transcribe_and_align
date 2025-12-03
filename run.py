from split_audio import split_audio_on_silence
from transcribe_audio import transcribe_chunks
from extract_pdf import pdf_to_txt


if __name__ == "__main__":
    audio_file = "/home/bo/workspace/whisper/tasks/HP1/audio_en/ch1.m4b"

    # extract text from PDF
    # for i in range(1, 8):
    #     p = f"/home/bo/workspace/whisper/tasks/HP1/text_zh/{i}.pdf"
    #     pdf_to_txt(p)

    # split audio
    # chunks_folder = split_audio_on_silence(
    #     audio_file,
    #     silence_ms=1000,  # split at pauses > 1s
    #     frame_ms=30,  # 30 ms VAD frames
    #     vad_mode=3,  # most sensitive
    #     min_chunk_sec=5.0,  # merge segments < 5 seconds
    # )

    # transcribe audio chunk
    #     chunks_folder = "/home/bo/workspace/whisper/tasks/HP1/audio_en/ch1_chunks"
    # transcribe_chunks(directory=chunks_folder, whisper_model="large")

   
    # align transcription with en and zh text
