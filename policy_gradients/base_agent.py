from abc import ABC, abstractmethod
import torch

"""
Policy based on if we are on-policy or off-policy
"""

class BaseAgent(ABC):
    def __init__(self,
                 env):
        super().__init__()
    
    def policy(self):
        pass

    @abstractmethod
    def train(self):
        pass