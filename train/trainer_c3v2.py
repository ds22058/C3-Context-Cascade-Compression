import os
import torch
import torch.nn as nn
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging, is_peft_available, is_safetensors_available

if is_peft_available():
    from peft import PeftModel

if is_safetensors_available():
    import safetensors.torch

logger = logging.get_logger(__name__)

TRAINING_ARGS_NAME = "training_args.bin"
SAFE_WEIGHTS_NAME = "model.safetensors"
WEIGHTS_NAME = "pytorch_model.bin"

IGNORE_INDEX = -100


def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    return model


class C3V2Trainer(Trainer):
    """Trainer for C3 v2 with separate saving of encoder (llm1) and decoder."""

    def __init__(self, decoder_lr_ratio: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.decoder_lr_ratio = decoder_lr_ratio

    # ------------------------------------------------------------------
    # Custom loss: per-task metrics (reconstruct / continuation)
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        task_types = inputs.pop("task_types", None)

        outputs = model(**inputs)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                f"NaN/Inf loss detected at step {self.state.global_step}, "
                f"replacing with 0 to skip this batch"
            )
            loss = loss.new_zeros(()) 
            loss.requires_grad_(True)
            return (loss, outputs) if return_outputs else loss

        if task_types is not None and self.state.global_step % self.args.logging_steps == 0:
            with torch.no_grad():
                logits = outputs.logits
                labels = inputs.get("labels")

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                per_token_loss = per_token_loss.view(shift_labels.size())

                valid_mask = shift_labels != IGNORE_INDEX
                per_sample_loss = (per_token_loss * valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1).clamp(min=1)

                recon_mask = task_types == 0
                cont_mask = task_types == 1

                metrics = {}
                if recon_mask.any():
                    metrics["train/loss_reconstruct"] = per_sample_loss[recon_mask].mean().item()
                    metrics["train/count_reconstruct"] = int(recon_mask.sum().item())
                if cont_mask.any():
                    metrics["train/loss_continuation"] = per_sample_loss[cont_mask].mean().item()
                    metrics["train/count_continuation"] = int(cont_mask.sum().item())

                valid_tokens = valid_mask.sum().item()
                total_tokens = shift_labels.numel()
                metrics["train/valid_token_ratio"] = valid_tokens / max(total_tokens, 1)
                metrics["train/seq_len_mean"] = inputs["input_ids"].shape[1]

                self.log(metrics)

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """Create optimizer with differential learning rates for Phase 2."""
        if self.optimizer is not None:
            return self.optimizer

        if self.decoder_lr_ratio >= 1.0:
            return super().create_optimizer()

        base_lr = self.args.learning_rate
        decoder_lr = base_lr * self.decoder_lr_ratio

        encoder_params = []
        decoder_params = []

        model = unwrap_model(self.model)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_encoder = any(
                key in name for key in ["llm1", "mm_projector", ".Q."]
            )
            if is_encoder:
                encoder_params.append(param)
            else:
                decoder_params.append(param)

        param_groups = [
            {"params": encoder_params, "lr": base_lr},
            {"params": decoder_params, "lr": decoder_lr},
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args, model
        )
        optimizer_kwargs.pop("lr", None)

        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        logger.info(
            f"Differential LR: encoder={base_lr}, decoder={decoder_lr} "
            f"(ratio={self.decoder_lr_ratio})"
        )
        logger.info(
            f"Encoder params: {len(encoder_params)}, Decoder params: {len(decoder_params)}"
        )
        return self.optimizer

    def _safe_save(self, output_dir: str):
        state_dict = self.model.state_dict()
        if self.args.should_save:
            cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
            del state_dict
            self._save(output_dir, state_dict=cpu_state_dict)
        else:
            del state_dict
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _save_part(self, output_dir, state_dict=None, model=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        supported_classes = (PreTrainedModel,)
        if is_peft_available():
            supported_classes = (PreTrainedModel, PeftModel)

        if not isinstance(model, supported_classes):
            if state_dict is None:
                state_dict = model.state_dict()
            unwrapped = self.accelerator.unwrap_model(model)
            if isinstance(unwrapped, supported_classes):
                unwrapped.save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors,
                )
            else:
                logger.info("Model is not PreTrainedModel, saving raw state dict.")
                if self.args.save_safetensors and is_safetensors_available():
                    safetensors.torch.save_file(
                        state_dict,
                        os.path.join(output_dir, SAFE_WEIGHTS_NAME),
                        metadata={"format": "pt"},
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
            )

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _save(self, output_dir=None, state_dict=None):
        if state_dict is None:
            state_dict = self.model.state_dict()

        state_dict_main = {k: v for k, v in state_dict.items() if "llm1" not in k}
        self._save_part(output_dir, state_dict=state_dict_main, model=self.model)

        model_unwrapped = unwrap_model(self.model)
        if hasattr(model_unwrapped, "model") and hasattr(
            model_unwrapped.model, "llm1"
        ):
            llm1_dir = os.path.join(output_dir, "llm1")
            self._save_part(
                llm1_dir, state_dict=None, model=model_unwrapped.model.llm1
            )
