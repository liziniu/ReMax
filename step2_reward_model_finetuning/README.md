# 🐕 Reward Model (RM) finetuning

This code is from [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning) with some modifications:

- Use the EOS token from the tokenizer as the PAD token.

- Incorporating the EOS token to the reward calculation in the reward model.

- Use only the last token loss in learning comparison data.