from abc import ABC, abstractmethod

class AbstractRunner(ABC):
    @abstractmethod
    def train_one_epoch(self):
        pass
    
    @abstractmethod
    def eval_model(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass
    
    @abstractmethod
    def model_summary(self):
        pass
