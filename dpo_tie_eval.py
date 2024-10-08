import argparse
import os, sys, torch
from copy import deepcopy
from os.path import join

import transformers
from accelerate import Accelerator
from torch.distributed import barrier
from tqdm import tqdm
from transformers import TrainingArguments
import yaml
from torch.nn.utils.rnn import pad_sequence
from trl.trainer.utils import DPODataCollatorWithPadding

from utils.dpo_tie_trainer_eval import DPOInference

sys.path.append('..')

from utils.utils import apply_chat_template
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, set_seed
from datasets import load_dataset
from loguru import logger

IGNORE_INDEX = -100

def create_tie_model(model_args):
    if model_args.tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer)
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id  # set as the <unk> token
    config = AutoConfig.from_pretrained(model_args.model)
    config._attn_implementation = "flash_attention_2"
    tokenizer.truncation_side = "left"
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model,
        config=config,
        trust_remote_code=True,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    if model_args.ref_model is not None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.ref_model,
            trust_remote_code=True,
            config=config,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else None, )
    else:
        ref_model = None

    return model, ref_model, tokenizer


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--ref_model", type=str, default=None, help="path to model")
    parser.add_argument(
        "--ref_free_type", type=str, default="avg", help="type of reference free normalization (norm, avg, or sum)"
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=6, help="batch size for inference")
    parser.add_argument("--dataset_path", type=str,
                        default="ultrafeedback_tied/test/non_tie_data_test.jsonl",
                        help="path of loss data")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--dpo_theta", type=float, default=-0.5, help="-alpha value fot ToDO")
    parser.add_argument("--dpo_beta", type=float, default=0.01, help="beta value for DPO and ToDO")
    parser.add_argument("--debug", type=bool, default=False, help="use only 10 examples")
    parser.add_argument("--original_reward", type=int, default=0, help="whether to use original reward")
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    set_seed(42)
    args = parser.parse_args()
    return args


BATCH_SIZE = 8


def main():
    accelerator = Accelerator()
    model_args = setup_everything()
    print("model_args.original_reward is ", model_args.original_reward)
    # return
    model, ref_model, tokenizer = create_tie_model(model_args)


    if model_args.ref_model is None:
        ref_free = True
    else:
        ref_free = False
    dataset = load_dataset("json", data_files=model_args.dataset_path, split="train")
    if model_args.debug:
        dataset = dataset.select(range(30))
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
        {"text_prompt": "prompt"})

    dpo = DPOInference(
        model=model,
        ref_model=ref_model,
        beta=model_args.dpo_beta,
        theta=model_args.dpo_theta,
        tokenizer=tokenizer,
        accelerator=accelerator,
        ref_free_norm=model_args.ref_free_type,
        # norm is norm, avg is average, sum is sum
    )
    # tokenize dataset
    column_names = list(dataset.features)
    print(column_names)
    tokenized_dataset = dataset.map(dpo.build_tie_batch, remove_columns=column_names)
    BATCH_SIZE=model_args.batch_size
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=dpo.label_pad_token_id,
            is_encoder_decoder=dpo.is_encoder_decoder,
        ),
        # collate_fn = lambda x: x, # fix weird batching error
        shuffle=False,
        drop_last=False,
    )
    results = []
    scores_chosen = []
    scores_tie = []
    scores_rejected = []
    tie_labels = []

    binary_results = []
    if model_args.original_reward==0:
        logger.info("start binary inference")
        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            logger.info(f"RM inference step {step}/{len(dataloader)}")
            rewards_chosen, reward_tie, rewards_rejected = dpo.inference_step(batch, ref_free=ref_free,
                                                                              tie_inference=False)

            # for each item in batch, record 1 if chosen > rejected
            # extra score from dict within batched results (e.g. logits)
            # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
            if isinstance(rewards_chosen[0], dict):
                scores_chosen_batch = [result["score"] for result in rewards_chosen]
                # scores_tie_batch = [result["tie"] for result in reward_tie]
                scores_rejected_batch = [result["score"] for result in rewards_rejected]
            # for classes that directly output scores (custom code)
            else:
                scores_chosen_batch = rewards_chosen.cpu().numpy().tolist()
                # scores_tie_batch = reward_tie.cpu().numpy().tolist()
                scores_rejected_batch = rewards_rejected.cpu().numpy().tolist()
            for item, chosen, rejected in zip(batch["tie"], scores_chosen_batch,
                                              scores_rejected_batch):
                if item == True:
                    continue
                else:
                    if chosen == max(chosen, rejected):
                        results.append(1)
                    else:
                        results.append(0)

            scores_chosen += scores_chosen_batch
            scores_rejected += scores_rejected_batch
            tie_labels += batch["tie"]

        print("evaluation acc")
        save_path = os.path.join(model_args.model, "original_valid_reward")
        assert save_path is not None
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        import json
        print("validation accuracy is : ", sum(results) / len(results))
        print("save data path is  : ", save_path + "/score_results.json")
        with open(os.path.join(save_path, "score_results.json"), 'w') as f:
            f.write(json.dumps(
                {"scores_chosen": scores_chosen, "scores_rejected": scores_rejected, "scores_tie": scores_tie,
                 "tie_labels": tie_labels, "results": results, "accuracy": sum(results) / len(results)}))
    elif  model_args.original_reward==1:
        logger.info("start tie inference")

        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            logger.info(f"RM inference step {step}/{len(dataloader)}")

            rewards_chosen, reward_tie, rewards_rejected = dpo.inference_step(batch, ref_free=ref_free,
                                                                              tie_inference=True, tie_type="tie_loss")

            # for each item in batch, record 1 if chosen > rejected
            # extra score from dict within batched results (e.g. logits)
            # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
            if isinstance(rewards_chosen[0], dict):
                scores_chosen_batch = [result["score"] for result in rewards_chosen]
                scores_tie_batch = [result["tie"] for result in reward_tie]
                scores_rejected_batch = [result["score"] for result in rewards_rejected]
            # for classes that directly output scores (custom code)
            else:
                scores_chosen_batch = rewards_chosen.cpu().numpy().tolist()
                scores_tie_batch = reward_tie.cpu().numpy().tolist()
                scores_rejected_batch = rewards_rejected.cpu().numpy().tolist()

            for item, chosen, tie, rejected in zip(batch["tie"], scores_chosen_batch, scores_tie_batch,
                                                   scores_rejected_batch):
                if item == True:
                    if tie == max(chosen, tie, rejected):
                        results.append(1)
                    else:
                        results.append(0)
                else:
                    if chosen == max(chosen, tie, rejected):
                        results.append(1)
                        binary_results.append(1)
                    else:
                        results.append(0)
                        binary_results.append(0)

            scores_chosen += scores_chosen_batch
            scores_rejected += scores_rejected_batch
            scores_tie += scores_tie_batch
            tie_labels += batch["tie"]
        save_path = os.path.join(model_args.model, "triple_valid_reward")
        assert save_path is not None
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        import json
        print("binary_accuracy is : ", sum(binary_results) / len(binary_results))
        print("validation accuracy is : ", sum(results) / len(results))
        print("save data path is  : ", save_path + "/score_results.json")
        with open(os.path.join(save_path, "score_results.json"), 'w') as f:
            f.write(json.dumps(
                {"scores_chosen": scores_chosen, "scores_rejected": scores_rejected, "scores_tie": scores_tie,
                 "tie_labels": tie_labels, "results": results, "accuracy": sum(results) / len(results)}))

if __name__ == "__main__":
    main()
