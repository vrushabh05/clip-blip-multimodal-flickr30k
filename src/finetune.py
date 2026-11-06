"""Four adaptation strategies for CLIP and BLIP on 10,000 Flickr30k images x 5 captions, 3 epochs each.

  linear   : CLIP projections only / BLIP LM head only          (lr 1e-3)
  partial  : last 2 transformer blocks (+ heads)                (lr 5e-5)
  lora     : LoRA r=16, alpha=32 on q/v (CLIP text encoder only; BLIP text decoder)  (lr 1e-4)
  full     : every parameter, mixed precision                   (CLIP lr 2e-5, BLIP lr 1e-5)

Usage: python finetune.py clip partial     |     python finetune.py blip full
"""
import sys

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor,
                          get_linear_schedule_with_warmup)

from data import BLIP_NAME, CLIP_NAME, DEVICE, Flickr30kFlat, load_split
from evaluate_models import caption_all, caption_metrics, embed_images, embed_texts, retrieval_metrics

which, method = sys.argv[1], sys.argv[2]
train, test, test_caps, queries = load_split()

if which == "clip":
    model, proc = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE), CLIPProcessor.from_pretrained(CLIP_NAME)
else:
    model, proc = BlipForConditionalGeneration.from_pretrained(BLIP_NAME).to(DEVICE), BlipProcessor.from_pretrained(BLIP_NAME)

BATCH = {("clip", "linear"): 128, ("clip", "partial"): 64, ("clip", "lora"): 64, ("clip", "full"): 32}
bs = BATCH.get((which, method), 32 if (which, method) == ("blip", "linear") else 16)
LR = {"linear": 1e-3, "partial": 5e-5, "lora": 1e-4, "full": 2e-5 if which == "clip" else 1e-5}[method]

if method != "full":
    for p in model.parameters():
        p.requires_grad = False
if method == "linear":
    heads = [model.visual_projection, model.text_projection] if which == "clip" else [model.text_decoder.cls]
    for h in heads:
        for p in h.parameters():
            p.requires_grad = True
elif method == "partial":
    if which == "clip":
        blocks = list(model.vision_model.encoder.layers[-2:]) + list(model.text_model.encoder.layers[-2:])
        heads = [model.visual_projection, model.text_projection]
    else:
        blocks, heads = list(model.text_decoder.bert.encoder.layer[-2:]), [model.text_decoder.cls]
    for part in blocks + heads:
        for p in part.parameters():
            p.requires_grad = True
elif method == "lora":
    if which == "clip":  # the vision encoder stays frozen: only the text side adapts
        cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none")
        model.text_model = get_peft_model(model.text_model, cfg)
        for h in (model.visual_projection, model.text_projection):
            for p in h.parameters():
                p.requires_grad = True
    else:
        cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "value"], lora_dropout=0.1, bias="none")
        model.text_decoder = get_peft_model(model.text_decoder, cfg)

trainable = [p for p in model.parameters() if p.requires_grad]
print(f"trainable {sum(p.numel() for p in trainable):,} / {sum(p.numel() for p in model.parameters()):,}")

dl = DataLoader(Flickr30kFlat(train, proc, mode=which), batch_size=bs, shuffle=True, num_workers=4,
                pin_memory=(which == "clip"))
opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, 300 if method == "full" else 200, 3 * len(dl)) \
    if (which == "clip" and method != "linear") or method == "full" else None
scaler = torch.cuda.amp.GradScaler() if method == "full" else None

model.train()
for epoch in range(3):
    total = 0.0
    for batch in tqdm(dl, desc=f"{which} {method} ep{epoch + 1}"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        opt.zero_grad()
        with torch.autocast("cuda", enabled=scaler is not None):
            loss = model(**batch, return_loss=True).loss if which == "clip" else model(**batch).loss
        if scaler:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            if method != "linear":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if sched:
            sched.step()
        total += loss.item()
    print(f"epoch {epoch + 1} loss {total / len(dl):.4f}")

model.save_pretrained(f"ckpt/{which}_{method}")
if which == "clip":
    print(retrieval_metrics(embed_texts(queries, model, proc), embed_images(test, model, proc))[0])
else:
    print(caption_metrics(caption_all(model, proc, test), [test_caps[i] for i in range(len(test))]))
