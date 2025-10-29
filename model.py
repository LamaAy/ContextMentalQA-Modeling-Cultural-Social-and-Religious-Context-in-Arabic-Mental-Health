import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class AraBERTMultiLabel(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self._pos_weight = None

    def set_pos_weight(self, pos_weight=None):
        self._pos_weight = pos_weight
        if pos_weight is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            h = outputs.pooler_output
        else:
            last_hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            h = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        logits = self.classifier(self.dropout(h))
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return logits, loss
