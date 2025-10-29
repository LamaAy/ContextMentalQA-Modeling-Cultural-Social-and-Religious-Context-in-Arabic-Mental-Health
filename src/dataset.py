from typing import List, Dict, Any
import ast
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def parse_labels_cell(x: str) -> List[str]:
    if x is None:
        return []
    x = str(x).strip()
    if x.startswith("[") and x.endswith("]"):
        try:
            out = ast.literal_eval(x)
            return [str(z).strip() for z in out]
        except Exception:
            pass
    if "|" in x:
        return [s.strip() for s in x.split("|") if s.strip()]
    if x == "" or x.lower() in {"no", "none", "[]"}:
        return []
    return [x]

class MentalQADataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name, max_len=192):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx].float()
        return item
