import requests
import pandas as pd
import os
import time, json, csv, re
from tqdm import tqdm

run_start_time = time.time()

audio_url = "./temp/_Fwf45pIAtM.wav"
audio_url = "./temp/t55mYdWC5mo.wav"

DEFAULT_SAVED_DIR = "transcription/gladia_transcription"
if not os.path.exists(DEFAULT_SAVED_DIR):
    os.mkdir(DEFAULT_SAVED_DIR)

headers = {
    "accept": "application/json",
    "x-gladia-key": "2b0f95c6-b864-4f0b-b83a-0ba1759d399e",
}

files = {
    "audio_url": (None, "https://www.youtube.com/watch?v=t55mYdWC5mo&t=441s"),
    "output_format": (None, "json"),
    "toggler_diarization": (None, "true"),
}

response = requests.post(
    "https://api.gladia.io/audio/text/audio-transcription/",
    headers=headers,
    files=files,
)


# print(response.text)
data = response.text
file_path = f"{DEFAULT_SAVED_DIR}/gl_transcribe.json"
csv_file_path = f"{DEFAULT_SAVED_DIR}/gl_transcribe.csv"
# json_data = json.dumps(data, indent=2)

with open(file_path, "w") as file:
    json.dump(data, file)


with open(file_path, "r") as file:
    data = file.read()

data = re.sub(r"\\", "", data.strip('"'))
data = json.loads(data)

# convert into csv

MIN_INTERVAL = 22
complete_message = ""
breakdown = list()
broken_message = ""
original_start_time = None

# words = []
# for prediction in data["prediction"]:
#     for word in prediction['words']:
#         words.append({"start": word["time_begin"], "end": word["time_end"], "text": word["word"]})
# df =pd.DataFrame(words)
# df.to_csv(csv_file_path, index=False)




new_df = pd.DataFrame(columns=["Start Time", "End Time", "Message"])

for each_breakdown in tqdm(data["prediction"]):
    complete_message += each_breakdown["transcription"]
    if original_start_time is None:
        original_start_time = each_breakdown["time_begin"]
    end_time = each_breakdown["time_end"]
    interval = end_time - original_start_time
    if interval > MIN_INTERVAL:
        breakdown.append(
            {
                "message": broken_message,
                "timestamp": f"{original_start_time}-{end_time}",
            }
        )

        new_df.loc[len(new_df)] = {
            "Start Time": original_start_time,
            "End Time": end_time,
            "Message": broken_message,
        }

        broken_message = ""
        original_start_time = None
        continue

    broken_message += each_breakdown["transcription"] + " "
new_df.to_csv(csv_file_path, index=False)
run_end_time = time.time()
print(f"Transcription Time: {run_end_time-run_start_time}")
