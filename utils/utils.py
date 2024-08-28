import os,sys
from typing import Literal
import torch
import torch.nn as nn
def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "dpo"],
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected", "tie")):
            if example["prompt"] is None:
                    raise ValueError("Prompt is empty")
            # For TPO, the inputs are four of (prompt, reference, chosen, rejected), where `reference` is gold response and `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            prompt_messages = [{"content": example['prompt'], "role": "user"}]
            # Prepend a system message if the first message is not a system message
            # if example["chosen"][0]["role"] != "system":
            prompt_messages.insert(0, {"role": "system", "content": ""})
            chosen_messages =  [{'content':example['chosen'], 'role': 'assistant'}]
            rejected_messages = [{'content':example['rejected'], 'role': 'assistant'}]
            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_tie"]=example["tie"]
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'tpo']}"
        )
    return example
PROMPT_TEMPLATE = dict(
    llama_alpaca=(
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: "
        ),
    llama2_alpaca=(
        "[INST] <<SYS>>\n"
        "You are a helpful assistant.\n"
        "<</SYS>>\n\n{instruction} [/INST]"
    ),
    default=(
        "Human: {instruction}\nAssistant: "
    )
)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)