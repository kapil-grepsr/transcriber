from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
from language_translator.base import BaseTranslator
import os


class MadladTranslator(BaseTranslator):

    def __init__(
            self,
            save_path: str = None,
            model_name: str = "google/madlad400-3b-mt",
            device: str = "cpu",

    ) -> None:
        self.model_name = model_name
        self.save_path = save_path
        self.device = device

    def _check_save_path(self) -> None:
        if self.save_path is None:
            if not os.path.exists(self.DEFAULT_SAVE_PATH):
                os.mkdir(self.DEFAULT_SAVE_PATH)
                self.save_path = self.DEFAULT_SAVE_PATH

            else:
                self.save_path = self.DEFAULT_SAVE_PATH
        
        else:
            if not os.path.exists(self.save_path):
                raise ValueError(f"Save path {self.save_path} does not exists")
            
    def translate(self, filename: str) -> str:
        self._check_save_path()
        
