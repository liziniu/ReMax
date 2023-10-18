# ReMax: A Simple, Effective, and Efficient Method for Aligning Large Language Models

## Overview

- This is the implementation of ReMax algorithm from the [paper](https://arxiv.org/abs/2310.10505). ReMax can be used in RLHF for aligning LLMs.


- Compared with the PPO algorithm in RLHF, ReMax is simple, efficient and effective. According to our practice, ReMax can saves about 50% GPU memory and it runs faster than PPO on large models.


<img src='./images/framework.png' width='600'>



- ReMax is easy to implement. We provide an implementation based on the DeepSpeed framework. You may explore more!


<img src='./images/algorithm.png' width='600'>



## How to use


### Prepare


The Python environment can be set up using Anaconda with the provided `environment.yml` file.

```
conda env create -f environment.yml
conda activate llm
```


### Step 1 SFT

```
cd step1_supervised_finetuning

# OPT(1.3B)
bash training_scripts/opt/run_opt_1.3b.sh

# Llama2(7B)
bash training_scripts/llama2/run_llama2_1.3b.sh
```

### Step 2 Reward Learning

```
cd step2_reward_model_finetuning

# OPT(1.3B)
bash training_scripts/opt/run_opt_1.3b.sh

# Llama2(7B)
bash training_scripts/llama2/run_llama2_1.3b.sh
```

### Step 3 RLHF

```
cd step3_rlhf_finetuning

# OPT(1.3B)
bash training_scripts/opt/run_opt_1.3b.sh

# Llama2(7B)
bash training_scripts/llama2/run_llama2_1.3b.sh
```


## Acknowledgements

Our code is heavily based on the [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat). Please follow the detailed instructions from DeepSpeed-Chat.


## Bibtex

If you find this code is helpful, please cite our paper in the following format.

```
@article{li2023remax,
  title     = {ReMax: A Simple, Effective, and Efficient Method for Aligning Large Language Models},
  author    = {Li, Ziniu and Xu, Tian and Zhang, Yushun and Yu, Yang and Sun, RUoyu and Luo, Zhi-Quan},
  booktitle = {arXiv preprint arXiv:2310.10505},
  year      = {2023},
}
```