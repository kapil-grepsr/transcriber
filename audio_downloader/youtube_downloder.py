import os
import librosa
import soundfile as sf
from audio_downloader.base import BaseDownloader
from pytube import YouTube


class YoutubeDownloader(BaseDownloader):

    def __init__(
        self, url: str, save_path: str = None, only_audio: str = "True"
    ) -> None:

        self.save_path = save_path

        self.yt = YouTube(url)

        self.yt_title = self.yt.title
        self.yt_id = self.yt.video_id

        self.only_audio = only_audio

    def _check_save_path(self) -> None:

        if self.save_path is None:

            if not os.path.exists(self.DEFAULT_SAVE_PATH):

                os.mkdir(self.DEFAULT_SAVE_PATH)

                self.save_path = self.DEFAULT_SAVE_PATH

            else:

                self.save_path = self.DEFAULT_SAVE_PATH

        else:

            if not os.path.exists(self.save_path):

                raise ValueError(f"Save path {self.save_path} does not exist")

    def _get_stream(self):

        stream = self.yt.streams.filter(only_audio=self.only_audio).first()

        return stream

    def _convert_to_wav(self, path):

        sound, sr = librosa.load(path)

        output_file = path.replace(".mp3", ".wav")

        print("Exporting", output_file)

        sf.write(output_file, sound, sr)

        os.remove(path)

        return output_file

    def download(self):

        self._check_save_path()

        stream = self._get_stream()

        filename = os.path.join(self.save_path, f"{self.yt_id}.mp3")

        stream.download(filename=filename)

        filename = self._convert_to_wav(filename)

        return filename
