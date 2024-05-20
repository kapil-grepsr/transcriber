from abc import ABC, abstractmethod


class BaseTranscriber(ABC):
    @abstractmethod
    def transcribe(self, filename: str) -> str:
        pass
