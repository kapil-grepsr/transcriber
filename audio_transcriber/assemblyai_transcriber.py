import assemblyai as aai
import time
import os
import csv
import pandas as pd

config_lang = aai.TranscriptionConfig(
    language_detection=True
)  # , disfluencies=True disfluencies is used not to remove filler words like "um", "uh"
aai.settings.api_key = "cede0a889bde41ba8812817f607b55a8"

audio_url = "./temp/_Fwf45pIAtM.wav"
audio_url = "./temp/t55mYdWC5mo.wav"


DEFAULT_SAVED_DIR = "transcription/assemblyai_transcription"
if not os.path.exists(DEFAULT_SAVED_DIR):
    os.mkdir(DEFAULT_SAVED_DIR)

csv_file_path = f"{DEFAULT_SAVED_DIR}/aai_transcribe.csv"
# csv_file_path = f"{DEFAULT_SAVED_DIR}/transcribe.csv"

assembly_transcription_start_time = time.time()

transcriber = aai.Transcriber(config=config_lang)

transcript = transcriber.transcribe(audio_url)
if transcript.error:
    print(transcript.error)
else:
    words = []
    data = transcript.words
    for word in data:
        words.append(
            {"start": word.start / 1000, "end": word.end / 1000, "text": word.text}
        )  # time is in milisecond
    df = pd.DataFrame(words)
    df.to_csv(csv_file_path, index=False)
    # print(transcript.words)


assembly_transcription_end_time = time.time()

print(
    f"Assembly Transcription Time: {assembly_transcription_end_time - assembly_transcription_start_time}"
)
