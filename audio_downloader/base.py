from abc import ABC, abstractmethod


class BaseDownloader(ABC):
    DEFAULT_SAVE_PATH = "temp"

    @abstractmethod
    def download(self) -> None:
        pass
