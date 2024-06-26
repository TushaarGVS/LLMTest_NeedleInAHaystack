from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from jsonargparse import CLI

from . import LLMNeedleHaystackTester, LLMMultiNeedleHaystackTester
from .evaluators import Evaluator, LangSmithEvaluator, OpenAIEvaluator, NumberEvaluator
from .providers import Anthropic, ModelProvider, OpenAI, Cohere, RecurrentGemma

load_dotenv()


@dataclass
class CommandArgs:
    provider: str = "openai"
    evaluator: str = "openai"
    model_name: str = "gpt-3.5-turbo-0125"
    evaluator_model_name: Optional[str] = "gpt-3.5-turbo-0125"
    needle: Optional[str] = (
        "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    )
    haystack_dir: Optional[str] = "PaulGrahamEssays"
    retrieval_question: Optional[str] = "What is the best thing to do in San Francisco?"
    results_version: Optional[int] = 1
    context_lengths_min: Optional[int] = 1000
    context_lengths_max: Optional[int] = 16000
    context_lengths_num_intervals: Optional[int] = 35
    context_lengths: Optional[list[int]] = None
    document_depth_percent_min: Optional[int] = 0
    document_depth_percent_max: Optional[int] = 100
    document_depth_percent_intervals: Optional[int] = 35
    document_depth_percents: Optional[list[float]] = None
    document_depth_percent_interval_type: Optional[str] = "linear"
    num_concurrent_requests: Optional[int] = 1
    save_results: Optional[bool] = True
    save_contexts: Optional[bool] = True
    final_context_length_buffer: Optional[int] = 200
    seconds_to_sleep_between_completions: Optional[float] = None
    print_ongoing_status: Optional[bool] = True
    # LangSmith parameters
    eval_set: Optional[str] = "multi-needle-eval-pizza-3"
    # Multi-needle parameters
    multi_needle: Optional[bool] = False
    needles: list[str] = field(
        default_factory=lambda: [
            " Figs are one of the secret ingredients needed to build the perfect pizza. ",
            " Prosciutto is one of the secret ingredients needed to build the perfect pizza. ",
            " Goat cheese is one of the secret ingredients needed to build the perfect pizza. ",
        ]
    )
    num_trials: int | None = None
    debug: bool | None = False


def get_model_to_test(args: CommandArgs) -> ModelProvider:
    """
    Determines and returns the appropriate model provider based on the provided command arguments.

    Args:
        args (CommandArgs): The command line arguments parsed into a CommandArgs dataclass instance.

    Returns:
        ModelProvider: An instance of the specified model provider class.

    Raises:
        ValueError: If the specified provider is not supported.
    """
    match args.provider.lower():
        case "openai":
            return OpenAI(model_name=args.model_name)
        case "anthropic":
            return Anthropic(model_name=args.model_name)
        case "cohere":
            return Cohere(model_name=args.model_name)
        case "recurrentgemma":
            # model_name: ["2b", "2b-it", "9b", "9b-it"]
            # evaluator: ["openai", "langsmith", "number"]
            return RecurrentGemma(
                model_name=args.model_name,
                evaluator_type=args.evaluator.lower(),
                debug=args.debug,
            )
        case _:
            raise ValueError(f"Invalid provider: {args.provider}")


def get_evaluator(args: CommandArgs) -> Evaluator:
    """
    Selects and returns the appropriate evaluator based on the provided command arguments.

    Args:
        args (CommandArgs): The command line arguments parsed into a CommandArgs dataclass instance.

    Returns:
        Evaluator: An instance of the specified evaluator class.

    Raises:
        ValueError: If the specified evaluator is not supported.
    """
    match args.evaluator.lower():
        case "openai":
            return OpenAIEvaluator(
                model_name=args.evaluator_model_name,
                question_asked=args.retrieval_question,
                true_answer=args.needle,
            )
        case "langsmith":
            return LangSmithEvaluator()
        case "number":
            return NumberEvaluator()
        case _:
            raise ValueError(f"Invalid evaluator: {args.evaluator}")


def main():
    """
    The main function to execute the testing process based on command line arguments.

    It parses the command line arguments, selects the appropriate model provider and evaluator,
    and initiates the testing process either for single-needle or multi-needle scenarios.
    """
    args = CLI(CommandArgs, as_positional=False)
    args.model_to_test = get_model_to_test(args)
    args.evaluator = get_evaluator(args)
    num_trials = args.num_trials if args.num_trials is not None else 1

    if args.multi_needle == True:
        print("Testing multi-needle")
        tester = LLMMultiNeedleHaystackTester(**args.__dict__)
    else:
        print("Testing single-needle")
        tester = LLMNeedleHaystackTester(**args.__dict__)

    for trial_num in range(num_trials):
        tester.start_test(trial_num=trial_num)


if __name__ == "__main__":
    main()
