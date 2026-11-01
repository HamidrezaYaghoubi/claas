"""Aggregate results/*.json from bench_claas.py into tables."""
import os
import re
import sys
import json
import glob
import statistics as st

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EVAL_ROWS = 2591  # survey rows inference_pred_llm.py loops over (33,683 responses / 13 per poster)


def ms(xs):
    return f"{st.mean(xs):6.2f} ± {st.stdev(xs) if len(xs) > 1 else 0:5.2f}"


def first_digit(s):
    m = re.search(r"\d", s)
    return m.group(0) if m else None


runs = {}
for p in sorted(glob.glob(os.path.join(RES, "*.json"))):
    r = json.load(open(p))
    runs[(r["workload"], r["variant"])] = r

for (wl, var), r in runs.items():
    R = r["records"]
    m = r["mem"]
    print(f"\n=== {wl} {var} | {r['gpu']} | {os.path.basename(r['model'])} | n={len(R)} | "
          f"fixed_len={r.get('fixed_len')} | load {r['load_sec']:.0f}s")
    print(f"  GPU memory: after load {m['after_load_gb']:.2f} GB, peak (nvidia-smi view) "
          f"{m['peak_device_used_gb']:.2f} GB, torch max reserved {m['torch_max_reserved_gb']:.2f} GB")
    print(f"  total per request (s):      {ms([x['total_sec'] for x in R])}")
    if wl == "demo":
        print(f"  first visible text (s):     {ms([x['first_visible_sec'] for x in R])}")
        print(f"  ... if A were streamed (s): {ms([x['first_visible_if_A_streamed_sec'] for x in R])}")
        for k in ["A_describe", "B_likert", "C_summary", "D_explain"]:
            line = f"  {k:12s} (s):          {ms([x[k]['sec'] for x in R])}"
            if "new_tokens" in R[0][k]:
                line += (f" | new tokens {ms([x[k]['new_tokens'] for x in R])} | prompt {R[0][k]['prompt_tokens']}"
                         f" | {st.mean(x[k]['new_tokens'] / x[k]['sec'] for x in R):5.1f} tok/s")
            if "prefix_tokens" in R[0][k]:
                line += f" | prefix {R[0][k]['prefix_tokens']} tok, suffixes {R[0][k]['suffix_tokens']}"
            print(line)
    else:
        print(f"  12 Likert (s):              {ms([x['likert']['sec'] for x in R])}", end="")
        if "prefix_tokens" in R[0]["likert"]:
            print(f" | prefix {R[0]['likert']['prefix_tokens']} tok, suffixes {R[0]['likert']['suffix_tokens']}")
        else:
            c = [cc for x in R for cc in x["likert"]["calls"]]
            print(f" | per call {ms([cc['sec'] for cc in c])} s, prompt {c[0]['prompt_tokens']} tok")
        if "free_text" in R[0]:
            print(f"  free text (s):              {ms([x['free_text']['sec'] for x in R])} | "
                  f"new tokens {ms([x['free_text']['new_tokens'] for x in R])}")
        tot = st.mean([x["total_sec"] for x in R])
        print(f"  projected full eval pass ({EVAL_ROWS} rows): {tot * EVAL_ROWS / 3600:.2f} h")


def agreement(wl, a, b):
    if (wl, a) not in runs or (wl, b) not in runs:
        return
    A = {(x["topic"], x["seed"]): x["answers"] for x in runs[(wl, a)]["records"]}
    B = {(x["topic"], x["seed"]): x["answers"] for x in runs[(wl, b)]["records"]}
    same = total = 0
    for k in A:
        for ga, sb in zip(A[k], B.get(k, [])):
            total += 1
            same += first_digit(ga) == first_digit(sb)
    print(f"Likert answers {wl}: {a} vs {b}: {same}/{total} identical")


print()
agreement("demo", "asis", "onemodel")
agreement("demo", "onemodel", "onemodel_prefix")
agreement("eval", "asis", "prefix")
