# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time

import deepspeed

# DeepSpeed Team
import torch
import torch.nn.functional as F
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from utils.utils import print_rank_0


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).cuda()
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f"{tag} {all_tensor}", rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = (
                hasattr(param, "ds_id")
                and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            )
            with deepspeed.zero.GatheredParameters(param, enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedReMaxTrainer:
    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token
        )["input_ids"][-1]
        self.z3_enabled = args.actor_zero_stage == 3
        self.z3_ref_enbale = args.reference_zero_stage == 3

        # Those value can be changed
        self.kl_ctl = args.kl_ctl
        self.clip_reward_value = 5.0
        self.gamma = args.gamma
        self.generate_time = 0.0

    def _generate_sequence(
        self,
        model,
        prompts,
        mask,
        step,
        print_answers=False,
        do_sample=True,
        synced_gpus=False,
        tag="model",
    ):
        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
        # if self.actor_model.module.config.model_type == "llama":
        #     kwargs = dict(do_sample=False)
        # else:
        #     kwargs = dict()
        kwargs = dict(
            do_sample=do_sample,
            top_p=0.9,
            temperature=1.0,
        )

        with torch.no_grad():
            seq = model.module.generate(
                prompts,
                attention_mask=mask,
                max_length=max_min_length,
                pad_token_id=self.tokenizer.pad_token_id,
                synced_gpus=synced_gpus,
                **kwargs,
            )

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt
        # without supervised fine tuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)

        if print_answers:
            print(
                f"[{tag}]--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
            )
            print(
                f"[{tag}]--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[i] <= 1:  # if the answer is shorter than 1 token, add " ".
                seq[i, self.prompt_length] = self.tokenizer.encode(" ")[-1]
            out_seq.append(seq[i : i + 1])
        out_seq = torch.cat(out_seq, dim=0)  # concat output in the batch dim

        return out_seq

    def generate_experience(
        self, prompts, mask, step, print_answers=False, training_mode=True
    ):
        self.eval()
        generate_start = time.time()
        seq = self._generate_sequence(
            self.actor_model,
            prompts,
            mask,
            step,
            print_answers,
            synced_gpus=self.z3_enabled,
        )
        if training_mode:
            baseline_seq = self._generate_sequence(
                self.actor_model,
                prompts,
                mask,
                step,
                print_answers,
                synced_gpus=self.z3_enabled,
                do_sample=False,
                tag="greedy",
            )
        generate_end = time.time()
        self.train()

        pad_token_id = self.tokenizer.pad_token_id

        action_mask = seq.not_equal(pad_token_id).long()
        if self.tokenizer.pad_token == self.tokenizer.eos_token:
            for i in range(seq.shape[0]):
                ans_mask = (
                    (seq[i, self.prompt_length :] == pad_token_id).nonzero().flatten()
                )
                if len(ans_mask) > 0:
                    # there exists an EOS token; we must set its action mask to be true.
                    # otherwise: the length may be increase
                    eos_token_pos = self.prompt_length + ans_mask[0].item()
                    action_mask[i, eos_token_pos] = 1

        if training_mode:
            baseline_action_mask = baseline_seq.not_equal(pad_token_id).long()
            if self.tokenizer.pad_token == self.tokenizer.eos_token:
                for i in range(baseline_seq.shape[0]):
                    ans_mask = (
                        (baseline_seq[i, self.prompt_length :] == pad_token_id)
                        .nonzero()
                        .flatten()
                    )
                    if len(ans_mask) > 0:
                        # there exists an EOS token
                        eos_token_pos = self.prompt_length + ans_mask[0].item()
                        baseline_action_mask[i, eos_token_pos] = 1

        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=action_mask)
            output_ref = self.ref_model(seq, attention_mask=action_mask)
            reward_score = self.reward_model.forward_value(
                seq, action_mask, prompt_length=self.prompt_length
            )["chosen_end_scores"].detach()

            if training_mode:
                baseline_reward_score = self.reward_model.forward_value(
                    baseline_seq, baseline_action_mask, prompt_length=self.prompt_length
                )["chosen_end_scores"].detach()

            values = torch.zeros_like(reward_score, device=reward_score.device)

        logits = output.logits
        logits_ref = output_ref.logits

        log_softmax_values = F.log_softmax(logits, dim=-1)
        softmax_probs = torch.exp(log_softmax_values)
        entropy = -torch.sum(softmax_probs * log_softmax_values, dim=-1)

        log_softmax_values_ref = F.log_softmax(logits_ref, dim=-1)
        full_kl = torch.sum(
            softmax_probs * (log_softmax_values - log_softmax_values_ref), dim=-1
        )

        logprobs = log_softmax_values.gather(
            dim=-1, index=seq[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        ref_logprobs = log_softmax_values_ref.gather(
            dim=-1, index=seq[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        self.generate_time = generate_end - generate_start

        return {
            "prompts": prompts,
            "logprobs": logprobs,
            "ref_logprobs": ref_logprobs,
            "value": values,
            "rewards": reward_score,
            "baseline_rewards": baseline_reward_score if training_mode else None,
            "full_kl": full_kl,
            "entropy": entropy,
            "input_ids": seq,
            "attention_mask": action_mask,
        }

    def compute_returns(self, prompts, kl_divergence, reward_score, action_mask):
        returns = torch.zeros_like(kl_divergence)
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1)  # + 1
        reward_clip = torch.clamp(
            reward_score, -self.clip_reward_value, self.clip_reward_value
        )
        batch_size = kl_divergence.shape[0]
        kl_ratio = 0.0
        count = 0
        for j in range(batch_size):
            cumulative_reward = reward_clip[j]
            cumulative_kl = 0
            for i in reversed(range(start, ends[j])):
                cumulative_kl = kl_divergence[j, i]

                cumulative_reward *= self.gamma
                returns[j, i] += cumulative_kl + cumulative_reward
                kl_ratio += torch.abs(cumulative_kl) / (
                    torch.abs(cumulative_reward) + torch.abs(cumulative_kl) + 1e-6
                )
                count += 1
        kl_ratio = kl_ratio / count
        return returns, kl_ratio

    def compute_loss(self, inputs):
        # train the rlhf mode here
        prompts = inputs["prompts"]
        log_probs = inputs["logprobs"]
        ref_log_probs = inputs["ref_logprobs"]
        reward_score = inputs["rewards"]
        baseline_reward_score = inputs["baseline_rewards"]
        attention_mask = inputs["attention_mask"]
        seq = inputs["input_ids"]

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        with torch.no_grad():
            kl_divergence = -(log_probs - ref_log_probs)
            kl_divergence = self.kl_ctl * kl_divergence

            reward_score = reward_score - baseline_reward_score
            returns, kl_ratio = self.compute_returns(
                prompts, kl_divergence, reward_score, action_mask
            )

        # process the new outputs
        batch = {"input_ids": seq, "attention_mask": attention_mask}
        logits = self.actor_model(**batch, use_cache=False).logits
        log_probs = gather_log_probs(logits[:, :-1, :], seq[:, 1:])

        actor_loss = self.actor_loss_fn(
            log_probs[:, start:], returns[:, start:], action_mask[:, start:]
        )
        return actor_loss, returns[:, start:], kl_ratio

    def get_overflow(self):
        actor_overflow = self.actor_model.optimizer.overflow
        # critic_overflow = self.critic_model.optimizer.overflow

        return actor_overflow

    def actor_loss_fn(self, logprobs, returns, mask):
        # policy gradient loss
        actor_loss = torch.sum(-returns * logprobs * mask) / mask.sum()
        return actor_loss

    def _validate_training_mode(self):
        assert self.actor_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()

    def eval(self):
        self.actor_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(
            f"{tag} global_actor_model_norm", actor_model_norm, self.args.local_rank
        )
        print_all_ranks(
            f"{tag} global_ref_model_norm", ref_model_norm, self.args.local_rank
        )
        print_all_ranks(
            f"{tag} global_reward_model_norm", reward_model_norm, self.args.local_rank
        )


class DeepSpeedReMaxTrainerUnsupervised(DeepSpeedReMaxTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
