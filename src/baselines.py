"""Zero-shot CLIP retrieval and BLIP captioning baselines."""
from transformers import BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor

from data import BLIP_NAME, CLIP_NAME, DEVICE, load_split
from evaluate_models import caption_all, caption_metrics, embed_images, embed_texts, retrieval_metrics

_, test, test_caps, queries = load_split()

clip, clip_proc = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE), CLIPProcessor.from_pretrained(CLIP_NAME)
m, ranks, _ = retrieval_metrics(embed_texts(queries, clip, clip_proc), embed_images(test, clip, clip_proc))
print("CLIP zero-shot", m)  # R@1 0.6157, R@5 0.8459, MRR 0.7211 in my run

blip, blip_proc = BlipForConditionalGeneration.from_pretrained(BLIP_NAME).to(DEVICE), BlipProcessor.from_pretrained(BLIP_NAME)
gen = caption_all(blip, blip_proc, test)
print("BLIP zero-shot", caption_metrics(gen, [test_caps[i] for i in range(len(test))]))
