from typing import List, Tuple

import librosa
from audio_transcriber.base import BaseTranscriber
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class WhisperTranscriber(BaseTranscriber):

    def __init__(
        self,
        audio_path: str = None,
        model_name: str = "openai/whisper-small",
        device: str = "cpu",
        hop_length: int = 700,
        frame_length: int = 700,
        chunk_size: int = 1024,
    ) -> None:

        self.model_name = model_name

        self.audio_path = audio_path

        self.device = device

        self.hop_length = hop_length

        self.frame_length = frame_length

        self.chunk_size = chunk_size

        self.processor = WhisperProcessor.from_pretrained(self.model_name)

        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)

        self.model.to(self.device)

        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )

    def _generate_stream(self) -> Tuple[object, int]:

        sr = librosa.get_samplerate(self.audio_path)

        stream = librosa.stream(
            self.audio_path,
            block_length=self.frame_length,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )

        return stream, sr

    def _chunkify_stream(
        self, stream: object, original_sr: int = None, target_sr: int = 16000
    ) -> Tuple[List[str], List[int], List[int]]:

        text = []

        start = []

        end = []

        duration = 0

        start_time = 0

        end_time = 0

        for sample in stream:

            start_time = end_time

            duration = int(len(sample) / original_sr)

            sample = librosa.resample(sample, orig_sr=original_sr, target_sr=target_sr)

            input_features = self.processor(
                sample,
                sampling_rate=target_sr,
                return_tensors="pt",
            ).input_features

            predicted_ids = self.model.generate(input_features.to(self.device))

            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                no_repeat_ngram_size=2,
            )

            text.append(transcription)

            end_time = start_time + duration

            start.append(start_time)

            end.append(end_time)

        return text, start, end

    def transcribe(self) -> Tuple[List[str], List[int], List[int]]:

        try:

            stream, sr = self._generate_stream()

            text, start, end = self._chunkify_stream(stream, sr)

            return text, start, end

        except FileNotFoundError as filenotfounderr:

            raise filenotfounderr

        except PermissionError as permerr:

            raise permerr

        except Exception as generalerr:

            raise generalerr
