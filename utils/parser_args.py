import sys, os, logging
import transformers
from transformers import HfArgumentParser, TrainingArguments
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    use_fast_tokenizer: Optional[bool] = field(default=False)
    dataset_path:str=field(default=None)
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    model_type: Optional[str] = field(
        default="auto",
        metadata={
            "help": (
                "If training from scratch, pass a model type from the list"
            ),
            "choices": ["auto","mistral"],
        },
    )
    load_in_4bit: bool = field(default=False)
    tie_data: bool = field(default=False)
    dpo_beta: float = field(default=0.1)
    trainable_theta: bool = field(default=False)
    max_length: int = field(default=512)
    low_rank_training: Optional[bool] = field(default=False)
    dpo_theta: Optional[float] = field(default=-0.2)
    loss_type: Optional[str] = field(
        default="sigmoid",
        metadata={
            "help": "dpo loss type"})
    max_prompt_length: Optional[int] = field(default=256)
    max_response_length: Optional[int] = field(default=256)
    min_response_length: Optional[int] = field(default=10)
    target_modules: List[str] = field(default_factory=lambda:["q_proj","k_proj","v_proj","o_proj"])
    lora_extra_params: List[str] = field(
        default_factory=lambda: ["embed_tokens", "norm"],
        metadata={"help": "Extra trainable parameters except LoRA weights, if low rank training."},
    )
    lora_rank: Optional[int] = field(default=8, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
    cache_dir: str = field(default="./cache_dir")
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    auto_insert_empty_system_msg: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to automatically insert an empty system message as the first message if `system` is mentioned in the chat template."
            )
        },
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})
    bnb_4bit_quant_storage: Optional[str] = field(
        default="uint8", metadata={"help": "storage type to pack the quanitzed 4-bit prarams."}
    )
