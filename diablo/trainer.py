"""Custom Trainer and TrainingArguments for DiaBlo."""

import os
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments


class CustomTrainingArguments(TrainingArguments):
    """Extended TrainingArguments with adapter-specific options."""

    def __init__(self, save_adapter_only: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.save_adapter_only = save_adapter_only


class CustomTrainer(Trainer):
    """Trainer that supports saving adapter-only checkpoints."""

    def create_optimizer(self):
        from torch.optim import AdamW
        main_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                main_params.append(param)
        self.optimizer = AdamW(
            main_params,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()

        if self.args.save_adapter_only:
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if ("block" in k or "score" in k)
            }
            torch.save(state_dict, os.path.join(output_dir, "adapter.chkpt"))
        else:
            torch.save(state_dict, os.path.join(output_dir, "model.chkpt"))

        torch.save(self.args, os.path.join(output_dir, "training_args.chkpt"))
