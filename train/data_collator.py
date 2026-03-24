import torch
import transformers

from train.config import IGNORE_INDEX


class C3V2DataCollator:
    """Pads input_ids, context_ids, labels to batch max length."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        context_ids = [inst["context_ids"] for inst in instances]
        task_types = torch.tensor(
            [inst.get("task_type", 0) for inst in instances], dtype=torch.long
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        context_ids = torch.nn.utils.rnn.pad_sequence(
            context_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            context_ids=context_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            context_attention_mask=context_ids.ne(self.tokenizer.pad_token_id),
            task_types=task_types,
        )
