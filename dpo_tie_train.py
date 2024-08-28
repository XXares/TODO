import argparse
import os, sys, torch
from copy import deepcopy
from os.path import join

import transformers
from torch.distributed import barrier
from transformers import TrainingArguments
import yaml
from torch.nn.utils.rnn import pad_sequence

sys.path.append('..')

from utils.parser_args import ModelArguments
from utils.utils import apply_chat_template
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, set_seed
from typing import Any, Dict, List, Optional
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from datasets import load_dataset
from loguru import logger

IGNORE_INDEX = -100
def create_model(model_args):
    ## load model
    if model_args.tokenizer_name_or_path is None:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id  # set as the <unk> token
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config._attn_implementation = "flash_attention_2"
    # if model_args.load_in_4bit:
    #     load_in_4bit= True
    #     quantization_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type="nf4"
    #     )
    #     model=model_class.from_pretrained(model_args.model_name_or_path,torch_dtype=torch.bfloat16,use_auth_token=True,quantization_config=quantization_config)
    tokenizer.truncation_side = "left"

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048


    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if model_args.low_rank_training:
        logger.info("Init new peft model")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=model_args.target_modules if model_args.target_modules else None,
            inference_mode=False,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )
        model = get_peft_model(model, peft_config=peft_config)
        model.print_trainable_parameters()

    else:
        logger.info("using full parameter training")

    return model, ref_model, tokenizer
def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='./train_args/mistral-7b-dpo.yaml', help="")
    parser.add_argument("--local_rank", type=int, default=0, help="")
    args = parser.parse_args()

    train_args_file = args.train_args_file
    # 读取训练的参数配置
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    model_args, training_args = parser.parse_yaml_file(yaml_file=train_args_file)
    # 创建输出目录
    rank = int(os.environ.get('RANK', -1))
    if rank == 0:
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
    logger.add(join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))
    # 加载训练配置文件
    with open(train_args_file, "r") as f:
        train_args = yaml.safe_load(f)
    # 保存训练参数到输出目录
    with open(join(training_args.output_dir, 'train_args.yaml'), "w") as f:
        yaml.dump(train_args, f)
    # 设置随机种子
    set_seed(training_args.seed)
    return model_args, training_args


def main():
    model_args, training_args = setup_everything()
    model, model_ref, tokenizer = create_model(model_args)

    dataset = load_dataset("json", data_files=model_args.dataset_path, split="train")
    column_names = list(dataset.features)
    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
        num_proc=8,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
        load_from_cache_file=False
    )
    dataset = dataset.rename_columns(
        {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected", "text_tie": "tie"})
    # print(dataset[0]["prompt"])
    # return
    # # Replace column names with what TRL needs, text_chosen -> chosen, text_rejected -> rejected, and text_reference -> reference
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=model_args.target_modules,
        inference_mode=False,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if model_args.loss_type not in ["tie_loss"]:
        from trl import DPOTrainer
        trainer = DPOTrainer(
            model=model,
            ref_model=model_ref,
            args=training_args,
            beta=model_args.dpo_beta,
            max_length=model_args.max_length,
            max_prompt_length=model_args.max_prompt_length,
            train_dataset=dataset,
            loss_type=model_args.loss_type,
            eval_dataset=None,
            peft_config=peft_config if model_args.low_rank_training else None,
            tokenizer=tokenizer)
    else:
        logger.info("using new dpo tie trainer")
        from utils.dpo_tie_trainer import DPOTieTrainer
        # print(model.get_dpo_theta())
        trainer = DPOTieTrainer(
            model=model,
            ref_model=model_ref,
            args=training_args,
            beta=model_args.dpo_beta,
            theta=model_args.dpo_theta,
            max_length=model_args.max_length,
            max_prompt_length=model_args.max_prompt_length,
            train_dataset=dataset,
            loss_type=model_args.loss_type,
            eval_dataset=None,
            peft_config=None,
            tokenizer=tokenizer,
        )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    print("save model")


if __name__ == "__main__":
    main()
