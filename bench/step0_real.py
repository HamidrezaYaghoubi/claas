"""
Step 0 of plan.md: the real trained adapter in the real classroom app.

Runs the app's own vlm_response() from the shared demo_task1_sph.py (loaded read-only via ast,
without gradio) on the 9 topic posters x 4 profiles, with the app's prompts and generation settings.
Only the objects behind `model_aux` / `tokenizer_aux` differ between the variants:

  asis      second 12B base model loaded for summary + explanation (exactly as the app's __main__)
  onemodel  the same 12B model with its LoRA adapter switched off for summary + explanation

Nothing else changes (no caching, no other optimisation); responses stop naturally (max 512).
"""
import os
import sys
import ast
import json
import time
import random
import hashlib
import argparse
import threading

SHARED = "/fs/nexus-projects/health_sim_ai"
APP = os.path.join(SHARED, "src", "demo_task1_sph.py")
ADAPTER = os.path.join(SHARED, "src", "unsloth", "gemma-3-12b-it_task1_1_epochs_test_neutral_partialTraits")
DATA = os.path.join(SHARED, "data", "survey_responses_screened.csv")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_SRC = os.path.join(HERE, "..", "src")  # utils.py here is byte-identical to the shared one
sys.dont_write_bytecode = True
sys.path.insert(0, REPO_SRC)

import unsloth  # noqa: F401  (before transformers)
from unsloth import FastVisionModel
from unsloth.chat_templates import get_chat_template
import torch
import pynvml
import yaml
import pandas as pd
from PIL import Image
from termcolor import colored
from transformers import TextIteratorStreamer
from utils import process_trait_info

# Profiles built only from the app's own dropdown values (checked below). P1 = the app's defaults.
PROFILES = [
    ["Female", "25–34", "Student", "White/Caucasian", "Leave Blank", "Leave Blank", "Leave Blank",
     "$75,000–$99,999", "Leave Blank"],
    ["Male", "45–54", "Healthcare/Medical", "Black/African American", "Christianity", "Moderate",
     "Bachelor’s degree", "$50,000–$74,999", "Married/Partnered, with children"],
    ["Non-binary/third gender", "18–24", "Unemployed/Looking for work", "Hispanic/Latino", "None of the above",
     "Liberal", "Some college, no degree", "Less than $25,000", "Single, living with family"],
    ["Female", "65 or older", "Retired", "Asian", "Buddhism", "Conservative", "Master’s degree",
     "$150,000 or more", "Married/Partnered, no children"],
]
FIELDS = ["Gender", "Age", "Current Profession", "Race/Ethnicity", "Religious/Cultural Group",
          "Political Affiliation", "Highest Education", "Annual Household Income", "Family Status"]
INPUT_KEYS = ("input_ids", "attention_mask", "token_type_ids", "pixel_values")
CALLS = []      # one record per generate() call of the current request
LIKERT_INPUTS = []  # processor inputs of every numerical-answer call (for the top-2 margin, after timing)
CAPTURED = []   # processor inputs of a few requests, for the adapter-switch check


def load_app_namespace():
    """Module-level constants + vlm_response() of the app, executed without its gradio UI."""
    tree = ast.parse(open(APP).read())
    keep = [n for n in tree.body if isinstance(n, ast.Assign) or
            (isinstance(n, ast.FunctionDef) and n.name == "vlm_response")]
    ns = {"os": os, "pd": pd, "Image": Image, "torch": torch, "random": random, "threading": threading,
          "colored": colored, "process_trait_info": process_trait_info}
    exec(compile(ast.Module(body=keep, type_ignores=[]), APP, "exec"), ns)
    return ns


def sha(t):
    return hashlib.sha1(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()[:16]


class GpuMem(threading.Thread):
    """Samples device memory used (what nvidia-smi shows) every 10 ms."""

    def __init__(self):
        super().__init__(daemon=True)
        pynvml.nvmlInit()
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        self.h = pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else "GPU-" + uuid)
        self.peak, self.stop = 0, False

    def used(self):
        return pynvml.nvmlDeviceGetMemoryInfo(self.h).used

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, self.used())
            time.sleep(0.01)


class Timed:
    """Stands in for `model` / `model_aux` in the app: times each generate() and records what it saw."""

    def __init__(self, m, tok, role, adapter_off, lora_layers):
        self._m, self._tok, self._role, self._off, self._lora = m, tok, role, adapter_off, lora_layers

    def __getattr__(self, k):
        return getattr(self._m, k)

    def _lora_off_fraction(self):
        if not self._lora:
            return None  # plain base model, no adapter at all
        return sum(bool(l.disable_adapters) for l in self._lora) / len(self._lora)

    def generate(self, **kw):
        n_in = kw["input_ids"].shape[1]
        rec = {"role": self._role, "max_new_tokens": kw.get("max_new_tokens"), "prompt_tokens": int(n_in),
               "input_ids_sha": sha(kw["input_ids"]),
               "pixel_sha": sha(kw["pixel_values"]) if kw.get("pixel_values") is not None else None}
        if self._role == "main" and kw.get("max_new_tokens") == 16:
            LIKERT_INPUTS.append({k: kw[k].to("cpu") for k in INPUT_KEYS if kw.get(k) is not None})
        # requests 1-3: keep the Likert, summary and explanation inputs (not the description)
        if CAPTURE[0] and not (self._role == "main" and kw.get("max_new_tokens") == 512):
            CAPTURED.append({"role": self._role, "max_new_tokens": kw.get("max_new_tokens"),
                             **{k: kw[k].to("cpu") for k in INPUT_KEYS if kw.get(k) is not None}})
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if self._off:
            with self._m.disable_adapter():
                rec["lora_off_fraction"] = self._lora_off_fraction()
                out = self._m.generate(**kw)
        else:
            rec["lora_off_fraction"] = self._lora_off_fraction()
            out = self._m.generate(**kw)
        torch.cuda.synchronize()
        rec["sec"] = time.perf_counter() - t0
        rec["new_tokens"] = int(out.shape[1] - n_in)
        rec["text"] = self._tok.batch_decode(out[:, n_in:])[0]
        CALLS.append(rec)
        return out


CAPTURE = [False]


def lora_layers(m):
    return [x for x in m.modules() if hasattr(x, "disable_adapters") and hasattr(x, "lora_A")]


def run_request(ns, topic, profile):
    CALLS.clear()
    poster = os.path.join(SHARED, "stimuli", ns["HEALTH_TOPICS"][topic] + ".png")
    user_input = {"text": "", "files": [poster]}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    first, last = None, ""
    for out in ns["vlm_response"](user_input, [], topic, *profile):
        if first is None:
            first = time.perf_counter() - t0
        last = out
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    # the app does not join its explanation thread; wait for that call's record before collecting
    deadline = time.time() + 30
    while sum(c["role"] == "aux" for c in CALLS) < 2 and time.time() < deadline:
        time.sleep(0.01)
    calls = list(CALLS)
    likert = [c for c in calls if c["max_new_tokens"] == 16]
    return {"topic": topic, "profile": PROFILES.index(profile), "total_sec": total, "first_yield_sec": first,
            "generate_sec": sum(c["sec"] for c in calls), "calls": calls,
            "likert_answers": [c["text"] for c in likert], "final_output": last}


def check_adapter_switch(model, base):
    """With both models loaded: adapter-off model vs the separately loaded base model, on real app inputs."""
    results = []
    for inp in CAPTURED:
        x = {k: v.to("cuda") for k, v in inp.items() if k in INPUT_KEYS}
        if "pixel_values" in x:
            x["pixel_values"] = x["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lb = base(**x, logits_to_keep=1).logits[0, -1].float()
            with model.disable_adapter():
                lo = model(**x, logits_to_keep=1).logits[0, -1].float()
            la = model(**x, logits_to_keep=1).logits[0, -1].float()
        gb = base.generate(**x, max_new_tokens=48, do_sample=False)
        with model.disable_adapter():
            go = model.generate(**x, max_new_tokens=48, do_sample=False)
        results.append({
            "role": inp["role"], "max_new_tokens": inp["max_new_tokens"],
            "off_vs_base_max_abs_logit_diff": float((lo - lb).abs().max()),
            "off_vs_base_same_top1": bool(lo.argmax() == lb.argmax()),
            "on_vs_base_max_abs_logit_diff": float((la - lb).abs().max()),
            "on_vs_base_same_top1": bool(la.argmax() == lb.argmax()),
            "off_vs_base_greedy48_identical": bool(torch.equal(gb, go)),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["asis", "onemodel"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ns = load_app_namespace()
    for prof in PROFILES:
        for field, value in zip(FIELDS, prof):
            assert value in ns["TRAIT_VALUES"][field], (field, value)
    cfgs = yaml.safe_load(open(os.path.join(SHARED, "src", "configs", "task1_demo_sph.yaml")))
    cfgs["data_path"] = DATA  # the app uses $HOME/data/... with HOME=/fs/nexus-projects/health_sim_ai
    ns["cfgs"] = cfgs
    ns["cfg_prompts"] = yaml.safe_load(open(os.path.join(SHARED, "src", "configs", "prompts.yaml")))

    mon = GpuMem()
    mem = {"idle_gb": mon.used() / 1e9}
    mon.start()
    t0 = time.perf_counter()
    # ---- same as the app's __main__ (demo_task1_sph.py:709-743), adapter path made absolute ----
    model, tokenizer = FastVisionModel.from_pretrained(model_name=ADAPTER, load_in_4bit=True)
    FastVisionModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    lora = lora_layers(model)
    if args.variant == "asis":
        model_aux, tokenizer_aux = FastVisionModel.from_pretrained(model_name=cfgs["model_summarize"],
                                                                   load_in_4bit=True)
        FastVisionModel.for_inference(model)  # as in the app (model_aux is never switched)
        tokenizer_aux = get_chat_template(tokenizer_aux, chat_template="gemma-3")
        aux = Timed(model_aux, tokenizer_aux, "aux", adapter_off=False, lora_layers=[])
    else:
        tokenizer_aux = tokenizer
        aux = Timed(model, tokenizer, "aux", adapter_off=True, lora_layers=lora)
    ns["model"] = Timed(model, tokenizer, "main", adapter_off=False, lora_layers=lora)
    ns["model_aux"], ns["tokenizer"], ns["tokenizer_aux"] = aux, tokenizer, tokenizer_aux
    ns["streamer"] = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    ns["streamer_aux"] = TextIteratorStreamer(tokenizer_aux, skip_prompt=True, skip_special_tokens=True)
    torch.cuda.synchronize()
    load_sec = time.perf_counter() - t0
    mem["after_load_gb"] = mon.used() / 1e9
    print(f"loaded in {load_sec:.1f}s, device used {mem['after_load_gb']:.2f} GB, LoRA layers: {len(lora)}",
          flush=True)

    topics = list(ns["HEALTH_TOPICS"])
    random.seed(0)
    torch.manual_seed(0)
    run_request(ns, "HIV", PROFILES[0])  # warm-up, not recorded
    torch.cuda.reset_peak_memory_stats()
    mon.peak = mon.used()

    records = []
    i = 0
    for topic in topics:
        for profile in PROFILES:
            i += 1
            CAPTURE[0] = i <= 3
            random.seed(i)
            torch.manual_seed(i)
            rec = run_request(ns, topic, profile)
            rec["seed"] = i
            records.append(rec)
            print(json.dumps({k: rec[k] for k in ("topic", "profile", "total_sec", "first_yield_sec",
                                                  "likert_answers")}, ensure_ascii=False), flush=True)
    CAPTURE[0] = False
    mem["peak_device_used_gb"] = mon.peak / 1e9
    mem["torch_max_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    mon.stop = True

    # how close the model's top two choices were for each numerical answer (after timing; adapter on)
    LIKERT_INPUTS[:3] = []  # drop the warm-up request
    assert len(LIKERT_INPUTS) == sum(len(r["likert_answers"]) for r in records)
    margins = []
    for inp in LIKERT_INPUTS:
        x = {k: v.to("cuda") for k, v in inp.items()}
        x["pixel_values"] = x["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            top = model(**x, logits_to_keep=1).logits[0, -1].float().topk(2)
        margins.append({"top1": tokenizer.tokenizer.decode(int(top.indices[0])),
                        "top2": tokenizer.tokenizer.decode(int(top.indices[1])),
                        "margin": float(top.values[0] - top.values[1])})
    for r in records:
        r["likert_margins"], margins = margins[:len(r["likert_answers"])], margins[len(r["likert_answers"]):]

    switch = check_adapter_switch(model, model_aux) if args.variant == "asis" else None
    out = {"variant": args.variant, "gpu": torch.cuda.get_device_name(0), "adapter": ADAPTER,
           "load_sec": load_sec, "mem": mem, "lora_layers": len(lora), "records": records,
           "adapter_switch_check": switch,
           "versions": {p: __import__(p).__version__ for p in
                        ["torch", "transformers", "unsloth", "peft", "bitsandbytes", "pandas"]}}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("MEM", json.dumps(mem), flush=True)
    if switch:
        print("SWITCH", json.dumps(switch), flush=True)


if __name__ == "__main__":
    main()
