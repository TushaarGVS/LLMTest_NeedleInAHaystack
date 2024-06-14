import re

from .evaluator import Evaluator


class NumberEvaluator(Evaluator):
    """https://arxiv.org/pdf/2406.04823."""

    @property
    def evaluator_type(self) -> str:
        return "number"

    def evaluate_response(self, response: str, **kwargs) -> int:
        true_number = kwargs["random_number"]
        # Convert all numbers to float to normalize x.0 and x. Also, replace all ','s to ensure x,xxx,xxx doesn't
        # get broken into multiple numbers.
        numbers_in_model_response = [
            float(s)
            for s in re.findall(r"[-+]?\d*\.\d+|\d+", response.replace(",", ""))
        ]
        # Ignoring the check on `len(numbers_in_model_response) == 1` to count even if model generates additional
        # (irrelevant) text beyond what is asked.
        return int(
            len(numbers_in_model_response) != 0
            and numbers_in_model_response[0].is_integer()
            and int(numbers_in_model_response[0]) == true_number
        )
