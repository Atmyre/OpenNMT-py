from abc import ABC, abstractmethod


class ReferenceLM(ABC):
    @abstractmethod
    def get_ppl(self, sentences):
        pass

