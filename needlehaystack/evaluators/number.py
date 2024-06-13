import re

from .evaluator import Evaluator


class NumberEvaluator(Evaluator):
    """https://arxiv.org/pdf/2406.04823."""

    @property
    def evaluator_type(self) -> str:
        return "number"

    def evaluate_response(self, response: str, **kwargs) -> int:
        true_number = kwargs["random_number"]
        numbers_in_model_response = [int(s) for s in re.findall(r"\b\d+\b", response)]
        return int(
            len(numbers_in_model_response) == 1
            and numbers_in_model_response[0] == true_number
        )
