"""Retrieval (Recall@1, Recall@5, MRR) and captioning (BLEU-4, ROUGE-L, METEOR, BERTScore) evaluation."""
import evaluate as hf_evaluate
import numpy as np
import torch

from data import DEVICE, IMG_COL


def _feat(model, **enc):
    f = model.get_image_features(**enc) if "pixel_values" in enc else model.get_text_features(**enc)
    return f if isinstance(f, torch.Tensor) else f.pooler_output


@torch.no_grad()
def embed_images(hf_data, model, proc, bs=64):
    model.eval()
    out = []
    for i in range(0, len(hf_data), bs):
        imgs = [x[IMG_COL].convert("RGB") for x in hf_data.select(range(i, min(i + bs, len(hf_data))))]
        f = _feat(model, **proc(images=imgs, return_tensors="pt", padding=True).to(DEVICE))
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out)


@torch.no_grad()
def embed_texts(texts, model, proc, bs=128):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = proc(text=texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=77).to(DEVICE)
        f = _feat(model, **enc)
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out)


def retrieval_metrics(txt, img):
    """Text-to-image retrieval: the correct image for query i is image i among all 1,551 test images."""
    sim = txt @ img.T
    ranks = np.array([(sim[i] > sim[i, i]).sum().item() + 1 for i in range(sim.shape[0])])
    return dict(R1=float(np.mean(ranks <= 1)), R5=float(np.mean(ranks <= 5)), MRR=float(np.mean(1.0 / ranks))), ranks, sim


@torch.no_grad()
def caption_all(model, proc, hf_data, bs=32):
    model.eval()
    out = []
    for i in range(0, len(hf_data), bs):
        imgs = [x[IMG_COL].convert("RGB") for x in hf_data.select(range(i, min(i + bs, len(hf_data))))]
        enc = proc(images=imgs, return_tensors="pt", padding=True).to(DEVICE)
        ids = model.generate(**enc, max_new_tokens=50, num_beams=4)
        out.extend(proc.batch_decode(ids, skip_special_tokens=True))
    return out


def caption_metrics(generated, gt_lists):
    """BLEU-4 uses all 5 reference captions; ROUGE-L, METEOR and BERTScore use the first reference only."""
    first = [g[0] for g in gt_lists]
    bleu = hf_evaluate.load("bleu").compute(predictions=generated, references=gt_lists)["bleu"]
    rouge = hf_evaluate.load("rouge").compute(predictions=generated, references=first)["rougeL"]
    meteor = hf_evaluate.load("meteor").compute(predictions=generated, references=first)["meteor"]
    bert = hf_evaluate.load("bertscore").compute(predictions=generated, references=first, lang="en")["f1"]
    return dict(BLEU4=round(bleu, 4), ROUGEL=round(rouge, 4), METEOR=round(meteor, 4), BERTScore=round(float(np.mean(bert)), 4))
