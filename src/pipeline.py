"""Retrieve-then-explain pipeline: CLIP retrieves the top 3 test images for a text query, BLIP captions each,
and a word-overlap heuristic scores relevance 1-5. The overlap score is a rough proxy, not a human judgement.
"""
import sys

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor

from data import BLIP_NAME, CLIP_NAME, DEVICE, IMG_COL, load_split
from evaluate_models import embed_images

CLIP_CKPT = sys.argv[1] if len(sys.argv) > 1 else CLIP_NAME  # e.g. ckpt/clip_partial
BLIP_CKPT = sys.argv[2] if len(sys.argv) > 2 else BLIP_NAME  # e.g. ckpt/blip_full

QUERIES = ["A child playing with a dog in a park", "A person cooking food in a kitchen",
           "A group of people hiking in the mountains", "A street scene with cars and pedestrians at night",
           "A person working on a laptop in a coffee shop"]
STOP = {"a", "an", "the", "in", "on", "at", "of", "with", "and", "is", "are", "to", "for"}


def relevance(query, caption):
    q, c = set(query.lower().split()) - STOP, set(caption.lower().split())
    ov = len(q & c) / max(len(q), 1)
    return 5 if ov >= 0.5 else 4 if ov >= 0.35 else 3 if ov >= 0.2 else 2 if ov >= 0.1 else 1


_, test, _, _ = load_split()
clip, cproc = CLIPModel.from_pretrained(CLIP_CKPT).to(DEVICE).eval(), CLIPProcessor.from_pretrained(CLIP_NAME)
blip, bproc = BlipForConditionalGeneration.from_pretrained(BLIP_CKPT).to(DEVICE).eval(), BlipProcessor.from_pretrained(BLIP_NAME)
index = embed_images(test, clip, cproc)

for q in QUERIES:
    with torch.no_grad():
        f = clip.get_text_features(**cproc(text=q, return_tensors="pt", padding=True, truncation=True).to(DEVICE))
        f = f if isinstance(f, torch.Tensor) else f.pooler_output
    sims = (f / f.norm(dim=-1, keepdim=True)).cpu() @ index.T
    print(f"\nQUERY: {q}")
    for rank, idx in enumerate(torch.argsort(sims[0], descending=True)[:3].tolist(), 1):
        enc = bproc(images=test[idx][IMG_COL].convert("RGB"), return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            cap = bproc.decode(blip.generate(**enc, max_new_tokens=50, num_beams=4)[0], skip_special_tokens=True)
        print(f"  rank {rank} sim {sims[0, idx]:.3f} relevance {relevance(q, cap)}/5  caption: {cap}")
