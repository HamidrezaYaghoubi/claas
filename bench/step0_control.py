"""
Control for the Step 0 adapter-switch check: is greedy generation repeatable on its own?
Same inputs as step0_real.py (app requests 1-3), both models loaded as in the app.
Compares 48-token greedy outputs: base vs base (repeat), adapter-off vs adapter-off (repeat),
and base vs adapter-off; reports where outputs first differ and how close the top-2 choices were there.
"""
import os
import json
import random
import torch
import yaml
from unsloth import FastVisionModel
from unsloth.chat_templates import get_chat_template
from transformers import TextIteratorStreamer
import step0_real as S


def greedy(m, x, off=False):
    kw = dict(**x, max_new_tokens=48, do_sample=False, output_scores=True, return_dict_in_generate=True)
    if off:
        with m.disable_adapter():
            return m.generate(**kw)
    return m.generate(**kw)


def compare(a, b, n_in):
    ta, tb = a.sequences[0, n_in:], b.sequences[0, n_in:]
    n = min(len(ta), len(tb))
    diff = (ta[:n] != tb[:n]).nonzero()
    if len(diff) == 0 and len(ta) == len(tb):
        return {"identical": True}
    i = int(diff[0]) if len(diff) else n
    top2 = a.scores[i][0].float().topk(2).values if i < len(a.scores) else None
    return {"identical": False, "first_diff_token": i,
            "top2_margin_at_diff": float(top2[0] - top2[1]) if top2 is not None else None}


def main():
    ns = S.load_app_namespace()
    cfgs = yaml.safe_load(open(os.path.join(S.SHARED, "src", "configs", "task1_demo_sph.yaml")))
    cfgs["data_path"] = S.DATA
    ns["cfgs"] = cfgs
    ns["cfg_prompts"] = yaml.safe_load(open(os.path.join(S.SHARED, "src", "configs", "prompts.yaml")))
    model, tokenizer = FastVisionModel.from_pretrained(model_name=S.ADAPTER, load_in_4bit=True)
    FastVisionModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    base, tok_base = FastVisionModel.from_pretrained(model_name=cfgs["model_summarize"], load_in_4bit=True)
    tok_base = get_chat_template(tok_base, chat_template="gemma-3")
    lora = S.lora_layers(model)
    ns["model"] = S.Timed(model, tokenizer, "main", adapter_off=False, lora_layers=lora)
    ns["model_aux"] = S.Timed(base, tok_base, "aux", adapter_off=False, lora_layers=[])
    ns["tokenizer"], ns["tokenizer_aux"] = tokenizer, tok_base
    ns["streamer"] = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    ns["streamer_aux"] = TextIteratorStreamer(tok_base, skip_prompt=True, skip_special_tokens=True)

    topics = list(ns["HEALTH_TOPICS"])
    S.CAPTURE[0] = True
    for i, (topic, profile) in enumerate([(topics[0], p) for p in S.PROFILES[:3]], start=1):
        random.seed(i)
        torch.manual_seed(i)
        S.run_request(ns, topic, profile)
    S.CAPTURE[0] = False

    rows = []
    for inp in S.CAPTURED:
        x = {k: v.to("cuda") for k, v in inp.items() if k in S.INPUT_KEYS}
        if "pixel_values" in x:
            x["pixel_values"] = x["pixel_values"].to(torch.bfloat16)
        n_in = x["input_ids"].shape[1]
        b1, b2 = greedy(base, x), greedy(base, x)
        o1, o2 = greedy(model, x, off=True), greedy(model, x, off=True)
        row = {"role": inp["role"], "max_new_tokens": inp["max_new_tokens"],
               "base_vs_base": compare(b1, b2, n_in), "off_vs_off": compare(o1, o2, n_in),
               "base_vs_off": compare(b1, o1, n_in)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_step0", "control.json")
    json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
