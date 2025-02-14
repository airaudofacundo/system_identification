from abc import ABC, abstractmethod

class dt_model(ABC):

    @abstractmethod
    def objective(self):
        pass

    @abstractmethod
    def gradient(self):
        pass
