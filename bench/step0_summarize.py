"""Summarize Step 0 (real adapter). Usage: step0_summarize.py DIR [DIR2 ...], each with asis.json / onemodel.json."""
import os
import sys
import json
import itertools
import statistics as st

DIRS = sys.argv[1:] or [os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_step0")]


def ms(xs):
    return f"{st.mean(xs):6.2f} ± {st.stdev(xs):5.2f} (median {st.median(xs):6.2f})" if len(xs) > 1 else f"{xs[0]:.2f}"


def stage_calls(rec):
    main512 = [c for c in rec["calls"] if c["role"] == "main" and c["max_new_tokens"] == 512]
    aux = [c for c in rec["calls"] if c["role"] == "aux"]
    out = {"A_describe": main512[:1], "B_likert": [c for c in rec["calls"] if c["max_new_tokens"] == 16],
           "C_summary": aux[:1], "D_explain": aux[1:2]}
    return {k: v for k, v in out.items() if v}


runs = {}
for d in DIRS:
    for v in ("asis", "onemodel"):
        p = os.path.join(d, f"{v}.json")
        if os.path.exists(p):
            runs[f"{os.path.basename(d.rstrip('/'))}/{v}"] = json.load(open(p))

for name, r in runs.items():
    R = r["records"]
    m = r["mem"]
    print(f"\n=== {name} | {r['gpu']} | n={len(R)} requests | load {r['load_sec']:.0f}s")
    print(f"  GPU memory: after load {m['after_load_gb']:.2f} GB, peak {m['peak_device_used_gb']:.2f} GB")
    print(f"  complete response (s): {ms([x['total_sec'] for x in R])}")
    print(f"  first text shown (s):  {ms([x['first_yield_sec'] for x in R])}")
    S = [stage_calls(x) for x in R]
    for k in ["A_describe", "B_likert", "C_summary", "D_explain"]:
        rows = [s[k] for s in S if k in s]
        offs = sorted({c["lora_off_fraction"] for row in rows for c in row}, key=str)
        sec = [sum(c["sec"] for c in row) for row in rows]
        tok = [sum(c["new_tokens"] for c in row) for row in rows]
        print(f"  {k:10s} n={len(rows):2d} sec {ms(sec)} | new tokens {st.mean(tok):6.1f} | "
              f"{st.mean(t / s for t, s in zip(tok, sec)):5.2f} tok/s | adapter-off fraction {offs}")
    if r.get("adapter_switch_check"):
        c = r["adapter_switch_check"]
        print(f"  adapter switch check on {len(c)} real app inputs: adapter-off vs base max|dlogit| = "
              f"{max(x['off_vs_base_max_abs_logit_diff'] for x in c)}, same top-1 {sum(x['off_vs_base_same_top1'] for x in c)}/{len(c)}, "
              f"greedy-48 identical {sum(x['off_vs_base_greedy48_identical'] for x in c)}/{len(c)}; "
              f"adapter-on vs base max|dlogit| between {min(x['on_vs_base_max_abs_logit_diff'] for x in c):.1f} and "
              f"{max(x['on_vs_base_max_abs_logit_diff'] for x in c):.1f}")

print("\n=== numerical answers: exact agreement between runs (same 36 requests x 3 questions)")
key = lambda x: (x["topic"], x["profile"])
for a, b in itertools.combinations(runs, 2):
    A = {key(x): x for x in runs[a]["records"]}
    B = {key(x): x for x in runs[b]["records"]}
    same = total = 0
    diffs = []
    for k in A:
        for i, (x, y) in enumerate(zip(A[k]["likert_answers"], B[k]["likert_answers"])):
            total += 1
            if x == y:
                same += 1
            else:
                ma = A[k].get("likert_margins", [None] * 3)[i]
                mb = B[k].get("likert_margins", [None] * 3)[i]
                diffs.append(f"{k[0]} / profile {k[1] + 1} / question {i + 1}: {x.replace('<end_of_turn>', '')} vs "
                             f"{y.replace('<end_of_turn>', '')} (top-2 margin {ma and round(ma['margin'], 3)} vs "
                             f"{mb and round(mb['margin'], 3)})")
    out_same = sum(A[k]["final_output"] == B[k]["final_output"] for k in A)
    print(f"  {a} vs {b}: answers {same}/{total} identical | full app output identical {out_same}/{len(A)}")
    for d in diffs:
        print("     differs:", d)

print("\n=== as-is vs one model, same run")
for d in DIRS:
    tag = os.path.basename(d.rstrip("/"))
    a, b = runs.get(f"{tag}/asis"), runs.get(f"{tag}/onemodel")
    if not (a and b):
        continue
    ma, mb = a["mem"]["peak_device_used_gb"], b["mem"]["peak_device_used_gb"]
    ta = st.mean(x["total_sec"] for x in a["records"])
    tb = st.mean(x["total_sec"] for x in b["records"])
    print(f"  {tag}: peak memory {ma:.2f} -> {mb:.2f} GB ({100 * (mb / ma - 1):+.1f}%), "
          f"complete response {ta:.2f} -> {tb:.2f} s ({100 * (tb / ta - 1):+.1f}%)")

flat = sorted(m["margin"] for r in runs.values() for rec in r["records"] for m in rec.get("likert_margins", []))
if flat:
    print(f"\n=== top-2 margin over all numerical answers with margins recorded (n={len(flat)}): "
          f"min {flat[0]:.3f}, 5th pct {flat[len(flat) // 20]:.3f}, median {flat[len(flat) // 2]:.3f}")
