from abc import ABC, abstractmethod


class Evaluator(ABC):
    CRITERIA: dict[str, str]

    @property
    def evaluator_type(self) -> str:
        return "evaluator_base"

    @abstractmethod
    def evaluate_response(self, response: str, **kwargs) -> int: ...
