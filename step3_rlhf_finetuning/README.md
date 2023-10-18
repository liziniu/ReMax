# 🐕 Reinforcement Learning from human feedback (RLHF) finetuning

This code is based on [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning) with some modifications:


- Use the [ReMax](https://arxiv.org/abs/2310.10505.pdf) algorithm to fine-tune LLMs

- Se the action mask to be 1 when generating an EOS token


