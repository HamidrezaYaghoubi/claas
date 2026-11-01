"""
Latency / GPU-memory benchmark of the CLAAS inference paths.

workload=demo : one poster request of src/demo_task1_sph.py (the classroom app)
                A) poster description  B) 3 Likert answers  C) summary  D) explanation
workload=eval : one survey sample of src/inference_pred_llm.py (12 Likert + 1 free text)

variants (demo)
  asis             fine-tuned 12B + a second 12B base for C/D, Likert by generate()
  onemodel         one 12B model, LoRA adapter switched off for C/D
  onemodel_prefix  onemodel + Likert by prefix-cached digit scoring
  onemodel_prefix_abase  onemodel_prefix + poster description (A) with the LoRA adapter off
  4b_prefix        onemodel_prefix with Gemma3-4B
variants (eval)
  asis             12 Likert generate() calls + 1 free-text generate()
  prefix           Likert by prefix-cached digit scoring, free text dropped (never scored)
  4b_prefix        prefix with Gemma3-4B
With --fixed_len (default) the demo's A/C/D generate exactly 90/60/160 tokens (the as-is
averages), so every variant does the same amount of decoding.
"""
import os
import sys
import time
import json
import random
import argparse
import threading

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

import unsloth  # noqa: F401  (must be imported before transformers)
from unsloth import FastVisionModel
from unsloth.chat_templates import get_chat_template
import torch
import pynvml
import yaml
from PIL import Image
from transformers import TextIteratorStreamer, HybridCache
from transformers.generation.streamers import BaseStreamer
from utils import process_trait_info

device = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
# Placeholder (all-zero LoRA) by default; point these at the real trained adapters for real-model runs.
ADAPTER = {
    "12b": os.environ.get("CLAAS_ADAPTER_12B", os.path.join(HERE, "adapters", "gemma-3-12b-it_task1_zeroLoRA")),
    "4b": os.environ.get("CLAAS_ADAPTER_4B", os.path.join(HERE, "adapters", "gemma-3-4b-it_task1_zeroLoRA")),
}
AUX_MODEL = "unsloth/gemma-3-12b-it"  # configs/task1_demo_sph.yaml: model_summarize

# ---------------------------------------------------------------------------
# Posters (real examples from the paper / slides) and survey questions.
# The real question wording is in the gated CSV; these follow the paper's examples
# and the demo's filters ("type in", "make you feel", "how open").
# ---------------------------------------------------------------------------
EMOTIONS = ["Angry", "Afraid", "Guilty", "Sad", "Ashamed", "Disgusted", "Hopeful", "Worried"]
TOPICS = {
    "Heart Disease": ("heart_disease_fear.png", "the health risks of heart disease",
                      "make healthy eating choices",
                      "an unhealthy diet to your heart health",
                      "adopting heart-healthy eating habits"),
    "Substance Abuse": ("substance_abuse_fear.png", "the health risks of substance abuse",
                        "not use substances", "substance use to your general health",
                        "trying a substance in the future"),
    "Mental Health": ("mental_health_efficacy.png", "the importance of my mental health",
                      "take care of my mental health",
                      "neglecting your mental health to your overall well-being",
                      "seeking help for mental health concerns"),
    "HIV": ("hiv_neutral.png", "the risks of HIV",
            "take steps to prevent HIV (e.g. testing, condoms, PrEP)",
            "HIV to a person's general health", "getting tested for HIV"),
}


def questions(topic):
    _, concern, motivate, harm, openq = TOPICS[topic]
    qs = [
        f"The message makes me more concerned about {concern}. - Scale: 1 (not at all) - 9 (extremely)",
        f"The message motivates me to {motivate} - Scale: 1 (not at all) - 9 (extremely)",
        f"In your opinion, how harmful is {harm}? - Scale: 0 (not at all harmful) - 6 (extremely harmful)",
        f"How open are you to {openq}? - Scale: 1 (not at all) - 9 (extremely)",
    ]
    qs += [f"To what extent did the material make you feel: Scale: 1 (not at all) - 9 (extremely) - {e}"
           for e in EMOTIONS]
    qs += ["Please type in your thoughts and reactions to this poster."]
    return qs  # 12 Likert + 1 free text = 13 per poster (paper, App. F)


def demo_questions(topic):
    out = []
    for q in questions(topic):
        ql = q.lower()
        if "type in" in ql or "make you feel" in ql or "how open" in ql:
            continue
        out.append(q)
    return out  # 3 questions, exactly the demo filter


def scale_digits(question):
    return list(range(0, 7)) if "0 (not at all" in question else list(range(1, 10))


# ---------------------------------------------------------------------------
# Persona inputs
# ---------------------------------------------------------------------------
def demo_persona():
    # demo_task1_sph.py dropdown defaults
    demo_dict = {
        "Gender": "Female", "Age": "25–34", "Current Profession": "Student",
        "Race/Ethnicity": "White/Caucasian", "Religious/Cultural Group": "Leave Blank",
        "Political Affiliation": "Leave Blank", "Highest Education": "Leave Blank",
        "Annual Household Income": "$75,000–$99,999", "Family Status": "Leave Blank",
    }
    demo_info = ""
    for trait, value in demo_dict.items():
        demo_info += f"{trait}: {value}\n" if value != "Leave Blank" else f"{trait}: [Not specified]\n"
    persona_score = "Big-Five Trait Scores:\n"
    demo_info, persona_score, _ = process_trait_info(
        demo_info, persona_score, None, demo_full=False, include_big5=True,
        include_facet=False, include_locus=False, train_mode=False)
    return demo_info


def eval_persona():
    # Approximation of one row of the gated survey CSV (full traits, as in
    # configs/task1_model_inference.yaml). Only its length matters here.
    demo_info = ("Demographics:\nGender: Female\nAge: 25–34\nCurrent Profession: Healthcare/Medical\n"
                 "Race/Ethnicity: White/Caucasian\nReligious/Cultural Group: Christianity\n"
                 "Political Affiliation: Moderate\nHighest Education: Bachelor’s degree\n"
                 "Annual Household Income: $75,000–$99,999\nFamily Status: Married/Partnered, with children\n")
    big5 = {"Extraversion": 3.25, "Agreeableness": 3.92, "Conscientiousness": 3.58,
            "Negative Emotionality": 2.67, "Open-Mindedness": 3.83}
    facets = {"Sociability": 3.0, "Assertiveness": 3.5, "Energy Level": 3.25, "Compassion": 4.0,
              "Respectfulness": 4.25, "Trust": 3.5, "Organization": 3.75, "Productiveness": 3.5,
              "Responsibility": 3.5, "Anxiety": 3.0, "Depression": 2.25, "Emotional Volatility": 2.75,
              "Intellectual Curiosity": 4.0, "Aesthetic Sensitivity": 3.5, "Creative Imagination": 4.0}
    persona_score = "\nBig-Five Trait Scores:\n" + "".join(f"{k}: {v}\n" for k, v in big5.items())
    persona_score += "Facet Scores:\n" + "".join(f"{k}: {v}\n" for k, v in facets.items())
    locus = ("an internal health locus of control score of 4.2 out of 6, a chance health locus of "
             "control score of 2.5 out of 6, and a powerful-others health locus of control score of 3.1 out of 6.")
    return process_trait_info(demo_info, persona_score, locus, True, True, True, True, True)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------
class GpuMem(threading.Thread):
    """Samples device memory used (what nvidia-smi shows) every 10 ms."""

    def __init__(self):
        super().__init__(daemon=True)
        pynvml.nvmlInit()
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        self.h = pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else "GPU-" + uuid)
        self.peak = 0
        self.stop_flag = False

    def used(self):
        return pynvml.nvmlDeviceGetMemoryInfo(self.h).used

    def run(self):
        while not self.stop_flag:
            self.peak = max(self.peak, self.used())
            time.sleep(0.01)

    def reset(self):
        self.peak = self.used()


class CountingStreamer(TextIteratorStreamer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.n_tokens = 0

    def put(self, value):
        if not (self.skip_prompt and self.next_tokens_are_prompt):
            self.n_tokens += value.numel()
        super().put(value)


class FirstTokenTimer(BaseStreamer):
    """Only timestamps the first generated token (generate() first puts the prompt)."""

    def __init__(self):
        self.calls, self.t_first = 0, None

    def put(self, value):
        self.calls += 1
        if self.calls == 2:
            self.t_first = time.perf_counter()

    def end(self):
        pass


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def clean(text, tok):
    eos = getattr(tok, "eos_token", None) or tok.tokenizer.eos_token
    return text.replace(eos, "").replace("<end_of_turn>", "")


def gen_once(model, tok, inputs, ttft=False, use_base=False, **kw):
    timer = FirstTokenTimer() if ttft else None
    t0 = sync()
    if use_base:
        with model.disable_adapter():
            out = model.generate(**inputs, use_cache=True, streamer=timer, **kw)
    else:
        out = model.generate(**inputs, use_cache=True, streamer=timer, **kw)
    t1 = sync()
    n_in = inputs.input_ids.shape[1]
    text = clean(tok.batch_decode(out[:, n_in:])[0], tok)
    stats = {"sec": t1 - t0, "prompt_tokens": n_in, "new_tokens": out.shape[1] - n_in}
    if ttft:
        stats["first_token_sec"] = timer.t_first - t0
    return text, stats


def gen_streamed(model, tok, inputs, use_base, **kw):
    """Same as the demo: generate() in a thread, consume a TextIteratorStreamer."""
    streamer = CountingStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    def target():
        if use_base:
            with model.disable_adapter():
                model.generate(**inputs, streamer=streamer, use_cache=True, **kw)
        else:
            model.generate(**inputs, streamer=streamer, use_cache=True, **kw)

    t0 = sync()
    th = threading.Thread(target=target)
    th.start()
    pieces, t_first = [], None
    for piece in streamer:
        if t_first is None and piece.strip():
            t_first = time.perf_counter()
        pieces.append(piece)
    th.join()
    t1 = sync()
    return "".join(pieces), {"sec": t1 - t0, "first_text_sec": (t_first or t1) - t0,
                             "prompt_tokens": inputs.input_ids.shape[1], "new_tokens": streamer.n_tokens}


def mm_inputs(tok, image, text):
    return tok(image.convert("RGB"), text, add_special_tokens=False, return_tensors="pt").to(device)


def chat(tok, content):
    return tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)


DIGIT_IDS = {}
CHECKED = []


def likert_prefix_cached(model, tok, image, prompts_text, qs):
    """The image + system + persona prefix is encoded once into a KV cache; each question then
    only runs its own suffix (question + instruction) against it. Answer = most likely valid digit."""
    if not DIGIT_IDS:
        for d in range(10):
            ids = tok.tokenizer.encode(str(d), add_special_tokens=False)
            assert len(ids) == 1, (d, ids)
            DIGIT_IDS[d] = ids[0]
    t0 = sync()
    first = tok(image.convert("RGB"), prompts_text[0], add_special_tokens=False, return_tensors="pt")

    def text_ids(t):
        # same steps as unsloth's patched Gemma3Processor.__call__ (unsloth_zoo/temporary_patches.py):
        # expand the image placeholder, drop the first "<bos>" text, tokenize
        t = t.replace(tok.boi_token, tok.full_image_sequence)
        i = t.find(tok.tokenizer.bos_token)
        t = t[i + len(tok.tokenizer.bos_token):] if i != -1 else t
        return tok.tokenizer(t, add_special_tokens=False).input_ids

    # same token ids the processor would produce, without re-processing the image per question
    ids = [first.input_ids[0]] + [torch.tensor(text_ids(t)) for t in prompts_text[1:]]
    if not CHECKED:
        assert text_ids(prompts_text[0]) == first.input_ids[0].tolist(), "manual tokenization differs from processor"
        CHECKED.append(True)
    P = min(len(x) for x in ids)
    for x in ids[1:]:
        diff = (x[:P] != ids[0][:P]).nonzero()
        if len(diff):
            P = int(diff[0])
    # +8 so the sliding-window layers never reach their last slot (HybridCache would roll the window)
    cache = HybridCache(config=model.config.get_text_config(), max_batch_size=1,
                        max_cache_len=max(len(x) for x in ids) + 8, device=device, dtype=torch.bfloat16)
    answers = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pre = ids[0][:P].unsqueeze(0).to(device)
        model(input_ids=pre, pixel_values=first.pixel_values.to(device, torch.bfloat16),
              attention_mask=torch.ones_like(pre), token_type_ids=first.token_type_ids[:, :P].to(device),
              past_key_values=cache, cache_position=torch.arange(P, device=device),
              use_cache=True, logits_to_keep=1)
        for x, q in zip(ids, qs):
            sfx = x[P:].unsqueeze(0).to(device)
            L = sfx.shape[1]
            out = model(input_ids=sfx, attention_mask=torch.ones(1, P + L, dtype=torch.long, device=device),
                        token_type_ids=torch.zeros_like(sfx), past_key_values=cache,
                        cache_position=torch.arange(P, P + L, device=device), use_cache=True, logits_to_keep=1)
            digits = scale_digits(q)
            scores = out.logits[0, -1, [DIGIT_IDS[d] for d in digits]].float()
            answers.append(str(digits[int(scores.argmax())]))
    t1 = sync()
    return answers, {"sec": t1 - t0, "prefix_tokens": P, "suffix_tokens": [len(x) - P for x in ids]}


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------
STYLE_VARIANTS = ["Write with a slightly informal and reflective tone.",
                  "Write in a straightforward conversational tone.",
                  "Write in a calm, matter-of-fact tone."]
LEXICAL = ["Feel free to vary sentence structures slightly.",
           "Use a light mix of paraphrasing expressions."]
OPENERS = ["This message", "From this message", "After seeing this message"]
OPENERS_POSTER = ["This poster", "My interpretation of the poster is", "To me, this poster is trying to"]
OPENERS_EXPLAIN = ["The reason why I think that is because", "My reasoning behind that is"]


FIXED_LEN = {"A": 90, "C": 60, "D": 160}  # as-is averages of 8 requests (job 7890418)


def lens(V, stage):
    if V["fixed_len"]:
        return {"max_new_tokens": FIXED_LEN[stage], "min_new_tokens": FIXED_LEN[stage]}
    return {"max_new_tokens": 512}


def demo_request(V, topic, prompts, demo_info):
    """Mirrors vlm_response() of demo_task1_sph.py, image-uploaded branch."""
    model, tok, aux, tok_aux = V["model"], V["tok"], V["aux"], V["tok_aux"]
    torch.cuda.empty_cache()  # the demo does this at the start of each request
    image = Image.open(os.path.join(HERE, "posters", TOPICS[topic][0]))
    style, lexical = random.choice(STYLE_VARIANTS), random.choice(LEXICAL)
    opening, opening_poster, opening_explain = (random.choice(OPENERS), random.choice(OPENERS_POSTER),
                                                random.choice(OPENERS_EXPLAIN))
    rec = {"topic": topic}
    t_req = sync()

    # A) poster description (fine-tuned model, not streamed). NOTE: PROMPT is a tuple, as in the demo.
    PROMPT = (
        f"Describe the content and main message in given heatlh campaign poster and how it's related to {topic}. ",
        "Note that the message could be non-direct or subtle (e.g. irony, fear-driven evoke without explicit texts, etc). Only provide the answer (in 2-4 sentences). ",
        f"Start the response with {opening_poster}",
    )
    inputs = mm_inputs(tok, image, chat(tok, [{"type": "image"}, {"type": "text", "text": PROMPT}]))
    image_desc, rec["A_describe"] = gen_once(model, tok, inputs, ttft=True, use_base=V["describe_base"],
                                             **lens(V, "A"),
                                             top_k=15, temperature=0.8, do_sample=True)

    # B) Likert answers
    SYSTEM_PROMPT = prompts["SYSTEM_SIM"]
    SIM_PROMPT = f"You are: Demographics:\n{demo_info}\n" + prompts["SIMULATION_SIM"]
    qs = demo_questions(topic)
    texts = [chat(tok, [{"type": "image"}, {"type": "text", "text": SYSTEM_PROMPT + SIM_PROMPT +
                         f"Question: {q}\n\n" + prompts["INSTRUCTION_MCQ"]}]) for q in qs]
    if V["likert"] == "generate":
        answers, per_q = [], []
        t0 = sync()
        for text in texts:
            a, s = gen_once(model, tok, mm_inputs(tok, image, text), max_new_tokens=16,
                            do_sample=False, temperature=0., min_p=0.9)
            answers.append(a)
            per_q.append(s)
        rec["B_likert"] = {"sec": sync() - t0, "calls": per_q}
    else:
        answers, rec["B_likert"] = likert_prefix_cached(model, tok, image, texts, qs)
    rec["answers"] = answers
    answers_numeric = "".join(f"{q}. Your answer: {a}\n" for q, a in zip(qs, answers))

    # C) summary (aux model, text only, streamed)
    USER_PROMPT = summary_prompt(topic, opening, style, lexical, answers_numeric)
    inputs = tok_aux(chat(tok_aux, [{"type": "text", "text": "You are a helpful assistant." + USER_PROMPT}]),
                     add_special_tokens=False, return_tensors="pt").to(device)
    response, rec["C_summary"] = gen_streamed(aux, tok_aux, inputs, V["use_base"], **lens(V, "C"),
                                              top_k=15, temperature=0.8, do_sample=True)

    # D) explanation + suggestions (aux model, with image, streamed)
    SIM2 = (f"You are: Demographics:\n{demo_info}\n" + prompts["SIMULATION_SIM"] +
            f"After seeing the uploaded impage, your response were {response}. "
            "Briefly explain WHY you responded that way, based on your demographic background. "
            f"Keep the explanation concise and direct. Start the response with '{opening_explain}' "
            f"(Style hint: {style}, concise; Lexical hint: {lexical}). "
            "Afterward, give a few *generic and succinct* suggestions to improve the poster's persuasiveness.")
    inputs = mm_inputs(tok_aux, image, chat(tok_aux, [{"type": "image"},
                                                      {"type": "text", "text": prompts["SYSTEM_SIM"] + SIM2}]))
    _, rec["D_explain"] = gen_streamed(aux, tok_aux, inputs, V["use_base"], **lens(V, "D"),
                                       min_p=0.85, temperature=0.1, do_sample=True)

    rec["total_sec"] = sync() - t_req
    # The user sees the first text when C starts streaming (A and B are not streamed).
    rec["first_visible_sec"] = (rec["A_describe"]["sec"] + rec["B_likert"]["sec"]
                                + rec["C_summary"]["first_text_sec"])
    # If A were streamed (it is the first block of the reply anyway), text would appear here:
    rec["first_visible_if_A_streamed_sec"] = rec["A_describe"]["first_token_sec"]
    rec["image_desc"], rec["summary"] = image_desc, response
    return rec


def summary_prompt(health_topic, opening_phrase, style_hint, lexical_hint, answers_numeric):
    # verbatim from demo_task1_sph.py
    return (
        "Summarize the following survey responses into a short, natural paragraph that captures your overall sentiment, motivation, and thinking. "
        f"Write as if paraphrasing what a person might say in conversation. Adjust your style based on your demographic/personality traits."
        "Do NOT repeat numeric scores. "
        "Preserve polarity: low scores → low concern/motivation/openness; high scores → high concern/motivation/openness. "
        "If answers are mixed (e.g., believes something is harmful but isn't personally moved), reflect that nuance explicitly. "
        "Keep to 1-5 sentences.\n\n"
        "**STRICTLY FOLLOW THESE RULES:**\n"
        "- Infer direction from each item's Scale description (e.g., 1-9: higher = more; 0-6: higher = more). "
        "- Use calibrated wording: 1-2 = very low, 3-4 = low, 5 = moderate, 6-7 = high, 8-9 = very high; for 0-6: 0-1 = not/slight, 2-3 = somewhat, 4-5 = high, 6 = very. "
        "- VERY IMPORTANT: provide ONLY the final summarized response, without anything else!"
        f"- The response MUST have a consistent health topic: {health_topic}. Ground each sentence to the impact of campaign message.\n"
        "- Never invert sentiment. Prefer hedged phrases (e.g., “not particularly,” “only somewhat,” “very open,” “not open at all”).\n\n"
        f"- Mimic the talking style of emulated demographic as realistic as possible."
        "**Example input 1:**\n"
        "The message makes me more concerned about the health risks of poor eating habits - Scale: 1-9. Your answer: 9\n"
        "The message motivates me to make healthy eating choices - Scale: 1-9. Your answer: 9\n"
        "In your opinion, how harmful is neglecting proper nutrition and weight management to your overall health? - Scale: 0–6. Your answer: 5\n"
        "How open are you to adopting healthier eating habits and lifestyle changes? - Scale: 1-9. Your answer: 9\n"
        "**Example output 1:**\n"
        "This message really heightened my awareness of how unhealthy eating can be. The content in the message strongly motivates me to make better choices, and I feel very ready to follow through.\n\n"
        "**Example input 2:**\n"
        "The message makes me more concerned about the health risks of COPD and smoking - Scale: 1-9. Your answer: 1\n"
        "The message motivates me to not smoke. - Scale: 1-9. Your answer: 1\n"
        "In your opinion, how harmful is smoking to your general health? - Scale: 0-6. Your answer: 6\n"
        "How open are you to smoking in the future? - Scale: 1-9. Your answer: 1\n"
        "**Example output 2:**\n"
        "From this message, I recognize smoking is very harmful, but the content in the message didn't increase my concern or motivate me much. It does somewhat make me understand that smoking is harmful, however. Anyway, I'm not open to smoking in the future.\n\n"
        "**Example input 3:**\n"
        "The message makes me more concerned about the effects of lack of exercise - Scale: 1-9. Your answer: 4\n"
        "The message motivates me to be more active - Scale: 1-9. Your answer: 3\n"
        "How open are you to exercising regularly? - Scale: 1-9. Your answer: 4\n"
        "**Example output 3:**\n"
        "Through the message, I get that exercise matters and the message raised my awareness a bit, but the poster content itself didn't really motivate me. The content in the message has some small impact in motivating me to change my routine.\n\n"
        f"Start the response with '{opening_phrase}'  (Style hint: {style_hint}; Lexical hint: {lexical_hint})\n"
        f"Input: {answers_numeric}. "
    )


def eval_sample(V, topic, prompts, persona):
    """Mirrors llm_simulate() of inference_pred_llm.py (unsloth engine, vision=True, trait=True)."""
    model, tok = V["model"], V["tok"]
    demo_info, persona_score, locus = persona
    image = Image.open(os.path.join(HERE, "posters", TOPICS[topic][0]))
    SYSTEM_PROMPT = prompts["SYSTEM_SIM"]
    SIM_PROMPT = (f"You are: {demo_info}\n"
                  f"Your personality test shows you have (min score = 0; max score = 5): {persona_score}\n"
                  f"You also have {locus}\n" + prompts["SIMULATION_SIM"])
    rec = {"topic": topic}
    t_req = sync()
    qs = questions(topic)
    likert_q = [q for q in qs if "type in" not in q.lower()]
    free_q = [q for q in qs if "type in" in q.lower()]

    def text_for(q, instr):
        return chat(tok, [{"type": "image"}, {"type": "text", "text": SYSTEM_PROMPT + SIM_PROMPT +
                          f"Question: {q}\n\n" + prompts[instr]}])

    if V["likert"] == "generate":
        calls, answers = [], []
        t0 = sync()
        for q in likert_q:
            a, s = gen_once(model, tok, mm_inputs(tok, image, text_for(q, "INSTRUCTION_MCQ")),
                            max_new_tokens=64, min_p=0.1, do_sample=False)
            answers.append(a)
            calls.append(s)
        rec["likert"] = {"sec": sync() - t0, "calls": calls}
    else:
        answers, rec["likert"] = likert_prefix_cached(model, tok, image,
                                                      [text_for(q, "INSTRUCTION_MCQ") for q in likert_q], likert_q)
    rec["answers"] = answers
    if V["free_text"]:
        _, rec["free_text"] = gen_once(model, tok, mm_inputs(tok, image, text_for(free_q[0], "INSTRUCTION_FREE")),
                                       max_new_tokens=64, min_p=0.1, do_sample=False)
    rec["total_sec"] = sync() - t_req
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load(name):
    model, tok = FastVisionModel.from_pretrained(model_name=name, load_in_4bit=True)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=["demo", "eval"], required=True)
    ap.add_argument("--variant", choices=["asis", "onemodel", "onemodel_prefix", "onemodel_prefix_abase",
                                          "prefix", "4b_prefix"],
                    required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--natural_len", action="store_true", help="let A/C/D stop at EOS (as in the demo)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = yaml.safe_load(open(os.path.join(SRC, "configs", "prompts.yaml")))
    mon = GpuMem()
    mem = {"idle_gb": mon.used() / 1e9}
    mon.start()

    size = "4b" if args.variant.startswith("4b") else "12b"
    t0 = time.perf_counter()
    model, tok = load(ADAPTER[size])
    FastVisionModel.for_inference(model)
    tok = get_chat_template(tok, chat_template="gemma-3")
    V = {"model": model, "tok": tok, "aux": model, "tok_aux": tok, "use_base": True,
         "likert": "generate" if args.variant in ("asis", "onemodel") else "prefix",
         "free_text": args.variant == "asis", "fixed_len": not args.natural_len,
         "describe_base": args.variant.endswith("_abase")}
    if args.variant == "asis" and args.workload == "demo":
        aux, tok_aux = load(AUX_MODEL)
        FastVisionModel.for_inference(model)  # as in the demo (aux is never switched)
        tok_aux = get_chat_template(tok_aux, chat_template="gemma-3")
        V.update(aux=aux, tok_aux=tok_aux, use_base=False)
    load_sec = time.perf_counter() - t0
    torch.cuda.synchronize()
    mem["after_load_gb"] = mon.used() / 1e9
    mem["torch_reserved_after_load_gb"] = torch.cuda.memory_reserved() / 1e9
    print(f"loaded in {load_sec:.1f}s; device used {mem['after_load_gb']:.2f} GB", flush=True)
    print("model class:", type(model).__name__, "| has disable_adapter:", hasattr(model, "disable_adapter"), flush=True)

    demo_info = demo_persona()
    persona = eval_persona()
    run = (lambda topic: demo_request(V, topic, prompts, demo_info)) if args.workload == "demo" \
        else (lambda topic: eval_sample(V, topic, prompts, persona))

    # warm-up (not recorded)
    random.seed(0)
    torch.manual_seed(0)
    run("HIV")
    torch.cuda.reset_peak_memory_stats()
    mon.reset()

    records = []
    for seed in range(1, args.seeds + 1):
        for topic in TOPICS:
            random.seed(seed)
            torch.manual_seed(seed)
            rec = run(topic)
            rec["seed"] = seed
            records.append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k not in ("image_desc", "summary")}), flush=True)

    mem["peak_device_used_gb"] = mon.peak / 1e9
    mem["torch_max_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    mem["torch_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    mon.stop_flag = True
    out = {"workload": args.workload, "variant": args.variant, "model": ADAPTER[size],
           "fixed_len": V["fixed_len"] and args.workload == "demo",
           "gpu": torch.cuda.get_device_name(0), "load_sec": load_sec, "mem": mem, "records": records,
           "versions": {"torch": torch.__version__, "unsloth": unsloth.__version__,
                        "transformers": __import__("transformers").__version__}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("MEM", json.dumps(mem), flush=True)


if __name__ == "__main__":
    main()
