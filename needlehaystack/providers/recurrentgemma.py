from pathlib import Path
from typing import Literal
from typing import Optional, Any

import kagglehub
import recurrentgemma
import sentencepiece as spm
import torch

from .model import ModelProvider


class RecurrentGemma(ModelProvider):
    def __init__(
        self,
        model_name: Literal["2b", "2b-it", "9b", "9b-it"],
        evaluator_type: Literal["number", "openai", "langsmith"],
        debug: bool = True,
        **model_kwargs: Any,
    ):
        self.debug = debug

        assert model_name in ["2b", "2b-it", "9b", "9b-it"]
        self.model_name = model_name
        assert evaluator_type in ["number", "openai", "langsmith"]
        self.evaluator_type = evaluator_type
        if self.debug:
            print(f"[DEBUG] evaluator type: {evaluator_type}")
            print(f"[DEBUG] loading recurrentgemma-{model_name} ...")

        self.__set_model_default_kwargs()
        for key, value in model_kwargs.items():
            self.__dict__.update({key: value})

        weights_dir = kagglehub.model_download(
            f"google/recurrentgemma/pyTorch/{model_name}"
        )
        weights_dir = Path(weights_dir)

        self.vocab = self._get_vocab_from_kaggle(weights_dir=weights_dir)
        self.model = self._get_recurrentgemma_from_kaggle(weights_dir=weights_dir)
        # Always set `is_it_model=False`, to avoid adding special tokens; we'll add them explicitly.
        self.sampler = recurrentgemma.Sampler(
            model=self.model,
            vocab=self.vocab,
            is_it_model=False,
            greedy_sampling=self.greedy_sampling,
        )
        if self.debug:
            print(f"[DEBUG] recurrentgemma-{model_name} loaded")
            print(f"[DEBUG] model config:\n{self.model.config}")

    def __set_model_default_kwargs(self):
        self.greedy_sampling = True
        self.max_tokens = 50
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _get_vocab_from_kaggle(weights_dir):
        vocab_path = weights_dir / "tokenizer.model"
        vocab = spm.SentencePieceProcessor()
        vocab.Load(str(vocab_path))
        return vocab

    def _get_recurrentgemma_from_kaggle(self, weights_dir):
        ckpt_path = weights_dir / f"{self.model_name}.pt"

        params = torch.load(str(ckpt_path))
        params = {key: value.to(device=self.device) for key, value in params.items()}
        preset = (
            recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1
            if "2b" in self.model_name
            else recurrentgemma.Preset.RECURRENT_GEMMA_9B_V1
        )
        model_config = recurrentgemma.GriffinConfig.from_torch_params(
            params, preset=preset
        )
        model = recurrentgemma.Griffin(
            model_config, device=self.device, dtype=torch.bfloat16
        )
        model.load_state_dict(params)

        return model

    def generate_prompt(self, context: str, retrieval_question: str) -> str:
        if self.evaluator_type == "number":
            prompt = f"""Some special magic number is hidden within the following articles. 
            Make sure to memorize it. 
            I will quiz you about the magic number afterwards.
            
            {context}
            
            Question: What is the special magic number mentioned in the provided text?
            Answer: The special magic number mentioned in the provided text is 
            """
        else:
            prompt = f"""You are a helpful AI bot that answers questions for a user.
            Make sure to memorize the contents of the following articles. 
            I will quiz you about the details in the articles afterwards.
            
            {context}

            Question: {retrieval_question} Keep your response short and direct. Don't include information outside 
            the articles or repeat your findings.
            Answer: 
            """
        if self.debug:
            print(f"[DEBUG] prompt: {prompt}")
        return prompt

    async def evaluate_model(self, prompt: str) -> str:
        model_output = self.sampler(
            input_strings=[prompt], total_generation_steps=self.max_tokens
        )
        model_response = model_output.text[0]
        if self.debug:
            print(f"[DEBUG] model response: {model_response}")
        return model_response

    def encode_text_to_tokens(self, text: str) -> list[int]:
        return self.sampler.tokenize(input_string=text).tolist()

    def decode_tokens(
        self, tokens: list[int], context_length: Optional[int] = None
    ) -> str:
        return self.vocab.DecodeIds(tokens[:context_length])
