"""
C3 v2 Dataset: supports reconstruct and continuation tasks.

Data format (JSONL):
  {"task": "reconstruct", "text": "..."}
  {"task": "continuation", "text": "...", "split_ratio": 0.55}
"""

import json
import copy
import random
import logging
import torch
from typing import Dict
from torch.utils.data import Dataset

from train.config import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    RECONSTRUCT_PROMPT,
    SYSTEM_MESSAGE,
    ROLE_USER,
    ROLE_ASSISTANT,
    SEP_TOKEN,
)

_SPECIAL_TOKENS_TO_SANITIZE = [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN]


class C3V2Dataset(Dataset):
    """Dataset for C3 v2 training with reconstruct and continuation tasks."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        latent_token_len: int = 32,
        model_max_length: int = 8192,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.latent_token_len = latent_token_len
        self.model_max_length = model_max_length

        self.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        self.im_start_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN])[0]
        self.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_END_TOKEN])[0]

        logging.warning(f"Loading data from {data_path} ...")
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        random.shuffle(self.data)
        logging.warning(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def _build_latent_placeholder(self) -> str:
        """Build the <img><imgpad>...<imgpad></img> placeholder string."""
        return (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_PATCH_TOKEN * self.latent_token_len
            + DEFAULT_IM_END_TOKEN
        )

    def _build_conversation(self, prompt_text: str, target_text: str) -> str:
        """Build MPT-style conversation string for decoder input."""
        placeholder = self._build_latent_placeholder()
        user_content = placeholder + ("\n" + prompt_text if prompt_text else "")
        return (
            SYSTEM_MESSAGE + SEP_TOKEN
            + ROLE_USER + user_content + SEP_TOKEN
            + ROLE_ASSISTANT + target_text + SEP_TOKEN
        )

    def _build_context(self, context_text: str) -> str:
        """Build context string for encoder input."""
        placeholder = self._build_latent_placeholder()
        return context_text + placeholder

    def _mask_labels(self, input_ids: torch.Tensor, conversation: str) -> torch.Tensor:
        """Mask everything except the assistant's response in labels."""
        labels = input_ids.clone()
        sep_plus_role = SEP_TOKEN + ROLE_ASSISTANT
        parts = conversation.split(sep_plus_role)
        if len(parts) < 2:
            labels[:] = IGNORE_INDEX
            return labels

        prefix = parts[0] + sep_plus_role
        prefix_len = len(self.tokenizer(prefix, add_special_tokens=False).input_ids)
        labels[:prefix_len] = IGNORE_INDEX
        return labels

    def _split_text_for_continuation(self, text: str, split_ratio: float):
        """Split text into context (front) and target (back) by token boundary."""
        tokens = self.tokenizer(text, add_special_tokens=False).input_ids
        split_idx = int(len(tokens) * split_ratio)
        split_idx = max(split_idx, 1)
        split_idx = min(split_idx, len(tokens) - 1)

        context_tokens = tokens[:split_idx]
        target_tokens = tokens[split_idx:]

        context_text = self.tokenizer.decode(context_tokens, skip_special_tokens=False)
        target_text = self.tokenizer.decode(target_tokens, skip_special_tokens=False)
        return context_text, target_text

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Remove special token strings that may appear in raw web data (e.g. HTML <img> tags)."""
        for tok in _SPECIAL_TOKENS_TO_SANITIZE:
            text = text.replace(tok, "")
        return text

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.data[i]
        task = item.get("task", "reconstruct")
        text = self._sanitize_text(item["text"])

        try:
            if task == "continuation":
                split_ratio = item.get("split_ratio", 0.55)
                context_text, target_text = self._split_text_for_continuation(
                    text, split_ratio
                )
                prompt = ""
            else:
                context_text = text
                target_text = text
                prompt = RECONSTRUCT_PROMPT

            context_str = self._build_context(context_text)
            conversation_str = self._build_conversation(prompt, target_text)

            input_ids = self.tokenizer(
                conversation_str,
                return_tensors="pt",
                max_length=self.model_max_length,
                truncation=True,
                add_special_tokens=False,
            ).input_ids[0]

            context_ids = self.tokenizer(
                context_str,
                return_tensors="pt",
                max_length=self.model_max_length,
                truncation=True,
                add_special_tokens=False,
            ).input_ids[0]

            labels = self._mask_labels(input_ids, conversation_str)

            if input_ids.shape[0] >= self.model_max_length:
                return self.__getitem__(random.randint(0, len(self.data) - 1))

            return dict(
                input_ids=input_ids,
                labels=labels,
                context_ids=context_ids,
                task_type=0 if task == "reconstruct" else 1,
            )

        except Exception:
            logging.warning(f"Error processing sample {i}, retrying with random sample.")
            return self.__getitem__(random.randint(0, len(self.data) - 1))
