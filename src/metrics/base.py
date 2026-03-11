from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseMetric(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, wrapper, sample: Dict[str, Any], xai_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Args:
            wrapper: The BaseVLMWrapper (for metrics that need to run the model).
            sample: The raw dictionary from the DataLoader (contains images, bboxes, questions).
            xai_result: The output from the Explainer (contains attributions, target_ids, generation time).
        """
        pass
