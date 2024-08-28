# ToDO: Tie rank introduced preference modeling makes better alignment for large language models

## Overview

Direct Preference Optimization (DPO) can significantly improve the performance of large language models (LLMs) on downstream tasks aligned with human preference. This process commonly applies the BT model and use preference datasets with pairwise preferred and dispreferred responses. However, if both responses are of high quality and there is no obvious preference difference between these responses, DPO can lead to sub-optimal preferences modeling and alignment results by treating such tied pairwise responses as "preferred'' and "dispreferred''. To address this issue, we introduce tie rank into current preference optimization procedure. We modify the BT model into the <u>**T**</u>ie rank introduced <u>**BT**</u> (TBT) model by introducing a threshold value of preference difference to fit the triplet preference relations. We then propose <u>**T**</u>ie rank intr<u>**o**</u>duced <u>**D**</u>irect Preference <u>**O**</u>ptimization (ToDO), an offline-policy method to improve preference modeling and alignment capacity of LLMs. We first analyse the limitations of DPO and advantages of ToDO in handling tie data theoretically. Then, we conduct experiments on both Mistral and Llama3 models to compare the effectiveness of DPO and ToDO. Experimental results demonstrate that ToDO, aligned with various ratios of tie data, achieves better preference modeling than DPO under both in-distribution and out-of-distribution data. Subsequently, we evaluate the alignment capacity of models aligned with DPO and ToDO on multiple benchmarks. Experimental results demonstrate that suitable ratio of tie data used in ToDO can improve model alignment capacity more effectively than DPO. In addition to being used in the ternary preference optimization process, ToDO can also be directly applied to binary preference alignment, achieving better alignment results than DPO. Further more, this tie rank introduced preference optimization procedure can not only be used in offline policies like DPO but can also be integrated into the reward model training process or other online optimization policies.

![framework](figs/framework.png)
## Model weights

## Set ups

```sh
conda create -n todo python=3.9 -y && conda activate todo
pip install -r requirements.txt
```

## Datasets

#### https://huggingface.co/datasets/irisxx/ultrafeedback_tied

## Usage and Examples

#### Training 

First, set up training config in a yaml. You can select any dataset used to conduct DPO/ToDO training.

##### DPO training

```shell
accelerate launch --config_file deep_zero3_config_process.yaml dpo_tie_train.py --train_args_file ./train_args/mistral-7b-dpo.yaml

accelerate launch --config_file deep_zero3_config_process.yaml dpo_tie_train.py --train_args_file ./train_args/llama3-8b-dpo.yaml
```

##### ToDO training

```shell
accelerate launch --config_file deep_zero3_config_process.yaml dpo_tie_train.py --train_args_file ./train_args/mistral-7b-todo.yaml

accelerate launch --config_file deep_zero3_config_process.yaml dpo_tie_train.py --train_args_file ./train_args/llama3-8b-todo.yaml
```

#### ToDO evalutation

##### Evaluation on test test

For DPO evaluation 

```shell
CUDA_VISIBLE_DEVICES=0,1 python3 dpo_tie_eval.py \
        --model ${policy_model_name_or_path} \ #policy model
        --ref_model ${reference_model_name_or_path}  \ #ref model
        --dataset_path ${test_set_path} \ #path of test set
        --original_reward 0 \ #0 for DPO and 1 for ToDO
        --dpo_beta 0.01 \ # beta value during alignment process
        --batch_size 8
```

For ToDO evaluation

```shell
CUDA_VISIBLE_DEVICES=0,1 python3 dpo_tie_eval.py \
        --model ${policy_model_name_or_path} \ #policy model
        --ref_model ${reference_model_name_or_path}  \ #ref model
        --dataset_path ${test_set_path} \ #path of test set
        --original_reward 1 \ #0 for DPO and 1 for ToDO
        --dpo_beta 0.01 \ # beta value during alignment process
        --dpo_theta -0.5 \ # default is -0.5, represents the -alpha value in ToDO, the same as training process
        --batch_size 8
```

##### Evalutaion of Reward Bench

We direct modify the implementation of Reward Bench to evaluate the preference modeling ability of DPO and ToDO, please refer to the implementation of Reward Bench
First, clone the repository of Reward Bench and set up the environment.


```shell

For DPO evaluation 

```shell
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/run_dpo.py ${policy_model_name_or_path} \
--ref_model ${reference_model_name_or_path} \
--dpo_beta 0.01 \
--evaluation_mode 0 \ #0 for DPO, 1 for ToDO
--save_path_prefix  Results/reward_bench_results \ #save path
--model_abbr ${set_model_abbr} \ # model abbreviation
--batch_size 8
#--prior_datsets 
```

For ToDO evaluation

```shell
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/run_dpo.py --model  ${policy_model_name_or_path} \
--ref_model ${reference_model_name_or_path} \
--dpo_theta -0.5 \
--dpo_beta 0.01 \ 
--evaluation_mode 1 \ #0 for DPO, 1 for ToDO
--save_path_prefix  Results/reward_bench_results/ \save path
--model_abbr ${set_model_abbr} \ # model abbreviation
--batch_size 8
#--prior_datsets 
```

#### Evalution on MT bench

We evaluate the performance follows the instruction of https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge

##### Noting that we  use gpt-4-turbo-2024-04-09 to score generated results. 

#### Evaluation of series popular benchmarks

We evalutaion the other benchmarks using opencompass . Pleaso first set up the environment required by opencompass follows the instruction https://github.com/open-compass/OpenCompass/ and we use the following prompt templates.

| Task            | piqa            | arc-c            | arc-e            | mmlu            | hellaswag            | winogrande           |
| --------------- | --------------- | ---------------- | ---------------- | --------------- | -------------------- | -------------------- |
| Prompt template | piqa_ppl_1cf9f0 | ARC_c_ppl_d52a21 | ARC_e_ppl_d52a21 | mmlu_ppl_ac766d | hellaswag_ppl_9dbb12 | winogrande_ll_c5cf57 |

Then you can evaluate the performance of popular benchmakrs using following shell instruction.

```shell
python run.py --models ${model_name_or_path} --datasets piqa_ppl ARC_c_ppl ARC_e_ppl mmlu_ppl hellaswag_ppl winogrande_ll
```



