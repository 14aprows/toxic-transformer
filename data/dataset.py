from datasets import load_dataset
import torch
from torch.utils.data import Dataset

class ToxicDataset(Dataset):
    def __init__(self, vocab, split="train", max_len=128):
        self.ds = load_dataset("jigsaw_toxicity_pred", split=split)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        item = self.ds[idx]
        text = item["comment_text"]

        ids = self.vocab.encode(text, self.max_len)
        input_ids = torch.tensor(ids)

        mask = (input_ids != 0).long()
        label = torch.tensor(int(item["toxic"] > 0.5))

        return input_ids, mask, label