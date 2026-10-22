

from abc import ABC, abstractmethod


class TrainingService(ABC):

    @abstractmethod
    def train(self, training_data):
        pass