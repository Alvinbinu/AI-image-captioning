# app.py
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import gradio as gr
import random
import math
import sys
import os

# ---- CONFIG ----
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
PARAPHRASER_MODEL = "t5-small"   # change to "google/flan-t5-base" later if desired
MAX_PARAPHRASE_LEN = 80  # used as max_new_tokens for paraphraser

# ---- device ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

# ---- helper to safely load models with clearer errors ----
def safe_from_pretrained(cls, *args, **kwargs):
    """
    Wrapper to call .from_pretrained with local_files_only=False and produce helpful error messages.
    """
    try:
        return cls.from_pretrained(*args, local_files_only=False, **kwargs)
    except Exception as e:
        print(f"\nERROR: failed to load {args[0] if args else 'model'}.")
        print("This often means your machine couldn't reach huggingface.co to download the model files.")
        print("Check your internet, DNS, or remove the cached huggingface files and retry.")
        print("Full error below:\n")
        raise

# ---- load BLIP (processor + model) ----
try:
    processor = safe_from_pretrained(BlipProcessor, BLIP_MODEL)
    model = safe_from_pretrained(BlipForConditionalGeneration, BLIP_MODEL).to(device)
    model.eval()
except Exception as e:
    # re-raise so user sees traceback (we printed earlier contextual guidance)
    raise

# ---- try to load paraphraser pipeline ----
paraphraser_available = False
try:
    # pipeline will also attempt to download the model if not present
    paraphraser = pipeline("text2text-generation", model=PARAPHRASER_MODEL, device=0 if device == "cuda" else -1)
    paraphraser_available = True
    print("Paraphraser loaded:", PARAPHRASER_MODEL)
except Exception as e:
    print("Paraphraser unavailable (will use template rewriting). Error:", e)
    paraphraser_available = False

# ---- template fallback banks ----
ADJECTIVES = [
    "ethereal", "golden", "serene", "mysterious", "vivid",
    "dreamlike", "tranquil", "melancholic", "radiant",
    "delicate", "bold", "graceful"
]
METAPHORS = [
    "a splash of sunlight", "a dancer frozen mid-twirl",
    "a quiet whisper", "a painting in motion",
    "a drifting dream", "a symphony of light"
]

def template_rewrite(caption: str, style="poetic"):
    adj = random.choice(ADJECTIVES)
    met = random.choice(METAPHORS)
    base = caption.strip().rstrip(".")
    if style == "poetic":
        return f"{base.capitalize()}, {adj} like {met}."
    if style == "vivid":
        return f"{adj.capitalize()} {base}."
    if style == "concise":
        return f"{base.capitalize()}."
    return f"{base.capitalize()}."

# ---- few-shot prompt templates (strict formatting) and paraphraser function ----
def paraphrase_with_t5(text: str, style="poetic", max_length=60, temperature: float = 0.7, top_p: float = 0.9, num_return_sequences: int = 1):
    """
    Few-shot prompt wrapper for the paraphraser. Uses 5 curated examples to teach style.
    Returns either a single string (if num_return_sequences==1) or a list of strings.
    """
    # If paraphraser unavailable, return template rewrites
    if not paraphraser_available:
        if num_return_sequences == 1:
            return template_rewrite(text, style)
        else:
            return [template_rewrite(text, style) for _ in range(num_return_sequences)]

    # Strict instruction + examples; instruct model to output ONLY the poetic line
    few_shot = """
You are a poetic caption writer. Your task:
- Read the simple factual caption.
- Output ONLY a single poetic rewritten line.
- DO NOT repeat the instructions or examples.
- DO NOT include anything except the final poetic line.

Examples:

Input: a group of jellyfish swimming in the water
Output: A drifting cluster of golden jellyfish, delicate and luminous like a slow painting beneath the waves.

Input: a sunlit street with people walking
Output: Sunlight threads through the crowd like molten ribbon, each step a quiet story.

Input: a mountain reflected in a lake
Output: The mountain sleeps upside-down on glassy water, a patient twin of blue.

Input: a portrait of a woman smiling
Output: Her smile is a small dawn, gentle and quietly contagious.

Input: a red bicycle leaning against a wall
Output: A red bicycle rests like a punctuation mark on the quiet street.

Now rewrite the caption below. Output ONLY the poetic line and nothing else.

Caption: "{text}"
""".strip()

    prompt = few_shot.replace("{text}", text)

    # Use max_new_tokens to avoid transformer warning about max_length vs max_new_tokens
    try:
        outputs = paraphraser(
            prompt,
            max_new_tokens=max_length,
            do_sample=(temperature > 0.0),
            temperature=float(temperature),
            top_p=float(top_p),
            num_return_sequences=int(num_return_sequences)
        )
    except TypeError:
        # Older pipeline versions may not accept max_new_tokens; fallback to max_length
        outputs = paraphraser(
            prompt,
            max_length=max_length,
            do_sample=(temperature > 0.0),
            temperature=float(temperature),
            top_p=float(top_p),
            num_return_sequences=int(num_return_sequences)
        )

    # Collect generated text(s)
    results = []
    if isinstance(outputs, list):
        for out in outputs:
            if isinstance(out, dict) and "generated_text" in out:
                results.append(out["generated_text"].strip())
    elif isinstance(outputs, dict) and "generated_text" in outputs:
        results.append(outputs["generated_text"].strip())

    # --- CLEANUP: keep only the last non-empty line of model output for each result ---
    cleaned = []
    for r in results:
        if not r:
            continue
        lines = [ln.strip() for ln in r.splitlines() if ln.strip()]
        # prefer last line (usually the final output), fallback to joined lines
        final = lines[-1] if lines else r.strip()
        # remove leading "Output:" or similar if accidentally included
        if final.lower().startswith("output:"):
            final = final.split(":", 1)[1].strip()
        cleaned.append(final)
    results = cleaned

    # If empty or model failed, fill with template rewrites
    while len(results) < num_return_sequences:
        results.append(template_rewrite(text, style))

    # Deduplicate while preserving order
    uniq = []
    seen = set()
    for r in results:
        if r not in seen:
            uniq.append(r)
            seen.add(r)

    if num_return_sequences == 1:
        return uniq[0] if uniq else template_rewrite(text, style)
    return uniq[:num_return_sequences]

# ---- compute confidence helper ----
def compute_confidence(sequence_scores):
    try:
        if sequence_scores is None:
            return None
        score = float(sequence_scores[0])
        prob = math.exp(score)
        prob = max(0.0, min(1.0, prob))
        return round(prob * 100, 1)
    except:
        return None

# ---- caption generation pipeline ----
def generate_caption(image: Image.Image, max_length: int = 50, num_beams: int = 5,
                     rewrite_style: str = "poetic", use_paraphraser: bool = True,
                     temperature: float = 0.7, top_p: float = 0.9, num_samples: int = 3):
    """
    Returns: chosen_styled (str), original_with_conf (str), alternatives (list[str])
    """
    if image is None:
        return "Please upload an image.", "No caption.", []

    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=int(max_length),
            num_beams=int(num_beams),
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True
        )

    # decode best sequence
    sequences = getattr(outputs, "sequences", None)
    if sequences is not None:
        raw_caption = processor.decode(sequences[0], skip_special_tokens=True)
    else:
        raw_caption = processor.decode(outputs[0], skip_special_tokens=True)

    conf = compute_confidence(getattr(outputs, "sequences_scores", None))
    original_info = f"Original: {raw_caption} (Confidence: {conf}%)" if conf is not None else f"Original: {raw_caption}"

    # generate paraphrases (list)
    if use_paraphraser and paraphraser_available:
        alts = paraphrase_with_t5(text=raw_caption, style=rewrite_style, max_length=MAX_PARAPHRASE_LEN,
                                  temperature=temperature, top_p=top_p, num_return_sequences=max(1, int(num_samples)))
    else:
        alts = [template_rewrite(raw_caption, rewrite_style) for _ in range(max(1, int(num_samples)))]

    # ensure list
    if isinstance(alts, str):
        alts = [alts]
    if not isinstance(alts, list):
        alts = list(alts)

    # final chosen caption = first alternative
    chosen = alts[0] if alts else template_rewrite(raw_caption, rewrite_style)
    return chosen, original_info, alts

# ---- wrapper to populate dropdown choices dynamically ----
def wrapper(image, max_caption_length, num_beams, style, use_paraphraser, temperature, top_p, num_samples):
    styled, orig, alts = generate_caption(image, max_caption_length, num_beams, style, use_paraphraser, temperature, top_p, num_samples)
    # return styled, orig, plus the list for the dropdown (Gradio 6 uses gr.update)
    return styled, orig, gr.update(choices=alts, value=alts[0] if alts else None)

# ---- Gradio UI (fn set to wrapper) ----
iface = gr.Interface(
    fn=wrapper,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=10, maximum=100, value=50, step=1, label="Max caption length"),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Num beams"),
        gr.Radio(choices=["poetic", "vivid", "concise"], value="poetic", label="Rewrite style"),
        gr.Checkbox(label="Use paraphraser (t5-small or flan-t5-base)", value=True),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="Temperature (creativity)"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p (nucleus sampling)"),
        gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Num samples (alternatives)")
    ],
    outputs=[
        gr.Textbox(label="Chosen Styled Caption"),
        gr.Textbox(label="Original Caption + Confidence"),
        gr.Dropdown(label="Alternative captions (pick one)", choices=[], type="value")
    ],
    title="AI Image Caption Generator (clean outputs)",
    description="BLIP base caption → few-shot paraphraser rewrites into poetic/vivid/concise styles. Use temperature and samples to control creativity."
)

if __name__ == "__main__":
    # bind only to localhost for safety and consistency
    iface.launch(server_name="127.0.0.1", server_port=7860)
