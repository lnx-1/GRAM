# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        tokenizer = kwargs["tokenizer"]

        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        if finetuning_args.gram_loss or finetuning_args.vgrm_loss:
            model = kwargs["model"]
            if finetuning_args.vgrm_loss and hasattr(model, "config"):
                model.config.output_hidden_states = True

            loss_name = "VGRM" if finetuning_args.vgrm_loss else "GRAM"
            logger.info_rank0(f"Replacing original SFT Loss with {loss_name} Loss ...")
            self.gram_candidate_labels = finetuning_args.gram_candidate_labels
            logger.info_rank0(f"GRAM candidate labels: {self.gram_candidate_labels}")
            self.gram_label_smoothing = finetuning_args.gram_label_smoothing
            logger.info_rank0(f"GRAM label smoothing: {self.gram_label_smoothing}")
            self.gram_candidate_labels_token_id = []
            for item in self.gram_candidate_labels:
                label_token_id = tokenizer(item, add_special_tokens=False).input_ids
                assert len(label_token_id) == 1, f"The number of token id for labels is greater than 1. Token:{item}, Token ID: {label_token_id}"
                self.gram_candidate_labels_token_id += label_token_id

            def get_label_positions_and_targets(labels, gram_candidate_labels_token_id):
                first_token_in_labels = [
                    tuple(torch.nonzero(label != IGNORE_INDEX)[0].tolist())[0]
                    for label in labels
                ]
                label_positions = torch.tensor(first_token_in_labels, device=labels.device)
                candidate_token_ids = torch.tensor(gram_candidate_labels_token_id, device=labels.device)
                label_token_ids = labels.gather(1, label_positions.unsqueeze(1)).squeeze(1)
                label_matches = label_token_ids.unsqueeze(1).eq(candidate_token_ids.unsqueeze(0))
                if not torch.all(label_matches.any(dim=1)):
                    raise ValueError(
                        f"Label token ids: {label_token_ids.tolist()}, "
                        f"Expected token ids: {gram_candidate_labels_token_id}"
                    )

                gram_labels = label_matches.float().argmax(dim=1).long()
                return label_positions, gram_labels, candidate_token_ids

            def gather_candidate_logits(outputs, labels, gram_candidate_labels_token_id):
                label_positions, gram_labels, candidate_token_ids = get_label_positions_and_targets(
                    labels, gram_candidate_labels_token_id
                )
                batch_indices = torch.arange(labels.size(0), device=labels.device)
                decision_positions = label_positions - 1
                logits = outputs.logits[batch_indices, decision_positions]
                return logits.index_select(dim=1, index=candidate_token_ids), gram_labels

            def project_candidate_logits(hidden_states, candidate_token_ids, output_embeddings):
                candidate_weights = output_embeddings.weight.index_select(
                    dim=0, index=candidate_token_ids.to(output_embeddings.weight.device)
                ).to(hidden_states.device)
                candidate_weights = candidate_weights.to(hidden_states.dtype)
                logits = torch.matmul(hidden_states, candidate_weights.transpose(0, 1))
                bias = getattr(output_embeddings, "bias", None)
                if bias is not None:
                    candidate_bias = bias.index_select(
                        dim=0, index=candidate_token_ids.to(bias.device)
                    ).to(hidden_states.device)
                    logits = logits + candidate_bias.to(logits.dtype)

                return logits

            def compute_gram_loss(outputs, labels, num_items_in_batch,
                                  gram_candidate_labels_token_id=self.gram_candidate_labels_token_id,
                                  label_smoothing=self.gram_label_smoothing):
                logits_candidate_tokens, gram_labels = gather_candidate_logits(
                    outputs, labels, gram_candidate_labels_token_id
                )
                return torch.nn.CrossEntropyLoss(
                    label_smoothing=label_smoothing
                )(
                    logits_candidate_tokens.float(),
                    gram_labels
                )

            def compute_vgrm_loss(outputs, labels, num_items_in_batch,
                                   gram_candidate_labels_token_id=self.gram_candidate_labels_token_id,
                                   label_smoothing=self.gram_label_smoothing,
                                   latent_dim=finetuning_args.vgrm_latent_dim,
                                   mc_samples=finetuning_args.vgrm_mc_samples,
                                   kl_weight=finetuning_args.vgrm_kl_weight,
                                   kl_warmup_steps=finetuning_args.vgrm_kl_warmup_steps,
                                   logvar_min=finetuning_args.vgrm_logvar_min,
                                   logvar_max=finetuning_args.vgrm_logvar_max):
                if outputs.hidden_states is None:
                    raise ValueError("VGRM loss requires `output_hidden_states=True` in the model config.")

                label_positions, gram_labels, candidate_token_ids = get_label_positions_and_targets(
                    labels, gram_candidate_labels_token_id
                )
                batch_indices = torch.arange(labels.size(0), device=labels.device)
                decision_positions = label_positions - 1
                decision_states = outputs.hidden_states[-1][batch_indices, decision_positions]
                hidden_size = decision_states.size(-1)
                if latent_dim * 2 > hidden_size:
                    raise ValueError(
                        f"`vgrm_latent_dim * 2` must be <= hidden size, got "
                        f"{latent_dim} * 2 > {hidden_size}."
                    )

                mu = decision_states[:, :latent_dim].float()
                logvar = decision_states[:, latent_dim: 2 * latent_dim].float().clamp(logvar_min, logvar_max)
                std = torch.exp(0.5 * logvar)
                num_samples = max(1, mc_samples)
                eps = torch.randn(
                    labels.size(0), num_samples, latent_dim, device=decision_states.device, dtype=torch.float32
                )
                sampled_z = mu.unsqueeze(1) + std.unsqueeze(1) * eps
                sampled_states = decision_states.float().unsqueeze(1).expand(-1, num_samples, -1).clone()
                sampled_states[:, :, :latent_dim] = sampled_z
                output_embeddings = model.get_output_embeddings()
                sampled_logits = project_candidate_logits(
                    sampled_states.to(decision_states.dtype), candidate_token_ids, output_embeddings
                ).float()
                expanded_labels = gram_labels.unsqueeze(1).expand(-1, num_samples).reshape(-1)
                preference_loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)(
                    sampled_logits.reshape(-1, sampled_logits.size(-1)),
                    expanded_labels,
                )
                kl_loss = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(dim=-1).mean()
                if kl_warmup_steps > 0:
                    current_step = max(0, getattr(getattr(self, "state", None), "global_step", 0))
                    kl_scale = min(1.0, float(current_step) / float(kl_warmup_steps))
                else:
                    kl_scale = 1.0

                return preference_loss + (kl_weight * kl_scale * kl_loss)

            kwargs["compute_loss_func"] = compute_vgrm_loss if finetuning_args.vgrm_loss else compute_gram_loss

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
