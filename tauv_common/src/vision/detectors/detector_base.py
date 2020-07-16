from abc import ABC, abstractmethod


class Detector(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_detection(self, src_image):
        pass
