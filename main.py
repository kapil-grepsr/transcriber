import pandas as pd
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from audio_downloader import YoutubeDownloader
from audio_transcriber import WhisperTranscriber
import time

DEFAULT_SAVED_DIR = "transcription"
if not os.path.exists(DEFAULT_SAVED_DIR):
    os.mkdir(DEFAULT_SAVED_DIR)


youtube_links = pd.read_csv("youtube_links.csv")
for link in tqdm(youtube_links["Links"]):
    download_start_time = time.time()
    downloader = YoutubeDownloader(url=link, only_audio="True")
    saved_path = downloader.download()
    download_end_time = time.time()
    transcription_start_time = time.time()
    transcriber = WhisperTranscriber(audio_path=saved_path)
    text, start, end = transcriber.transcribe()
    data = {"start":start, "end":end, "text":[x[0] for x in text]}
    df = pd.DataFrame(data)
    df.to_csv(f"{DEFAULT_SAVED_DIR}/{Path(saved_path).name.replace(".wav", ".csv")}", index=False)
    transcription_end_time=time.time()

print(f"Download Time {download_end_time-download_start_time}")
print(f"Transcription Time{transcription_end_time-transcription_start_time}")





    
