from abc import ABC, abstractmethod

class BaseTranslator(ABC):
    DEFAULT_SAVE_PATH = "translation"

    @abstractmethod
    def translate(self, filename: str) -> str:
        pass