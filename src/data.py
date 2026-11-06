"""Flickr30k loading with a fixed 95/5 image-level split, plus a dataset that flattens 5 captions per image."""
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_NAME = "openai/clip-vit-base-patch32"
BLIP_NAME = "Salesforce/blip-image-captioning-base"
IMG_COL, CAP_COL = "image", "original_alt_text"  # columns of Mozilla/flickr30k-transformed-captions


def load_split(seed=42, test_size=0.05, n_train_images=10000):
    ds = load_dataset("Mozilla/flickr30k-transformed-captions", split="test")  # all 31,014 images live in "test"
    train_idx, test_idx = train_test_split(list(range(len(ds))), test_size=test_size, random_state=seed, shuffle=True)
    train, test = ds.select(train_idx), ds.select(test_idx)  # 29,463 train / 1,551 test images
    train_subset = train.select(range(min(n_train_images, len(train))))  # fine-tuning uses the first 10,000
    test_caps = {i: list(test[i][CAP_COL]) for i in range(len(test))}
    queries = [test[i][CAP_COL][0] for i in range(len(test))]  # retrieval query = first caption of each image
    return train_subset, test, test_caps, queries


class Flickr30kFlat(Dataset):
    """Every (image, caption) pair: 10,000 images x 5 captions = 50,000 training pairs."""

    def __init__(self, hf_data, processor, mode="clip", max_len=77):
        self.data, self.processor, self.mode, self.max_len = hf_data, processor, mode, max_len
        self.pairs = [(i, j) for i in range(len(hf_data)) for j in range(5)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        item = self.data[i]
        image, caption = item[IMG_COL].convert("RGB"), item[CAP_COL][j]
        if self.mode == "clip":
            enc = self.processor(text=caption, images=image, return_tensors="pt", padding="max_length",
                                 truncation=True, max_length=self.max_len)
        else:
            enc = self.processor(images=image, text=caption, return_tensors="pt", padding="max_length",
                                 truncation=True, max_length=self.max_len)
            enc["labels"] = enc["input_ids"].clone().squeeze(0)
        return {k: v.squeeze(0) for k, v in enc.items()}
