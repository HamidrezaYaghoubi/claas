# Plan: making the CLAAS classroom app more deployable

**Goal:** make the classroom app (`src/demo_task1_sph.py`) cheaper and faster to serve without changing what the model outputs.
Steps are ordered from "intended to keep outputs unchanged" to "outputs may change". Each step is validated with the real trained adapter before it is kept.

## Where we are today (measured with the real model)

One poster request on an RTX A6000, model time only (no browser, network or queue time):

| | |
|---|---|
| Complete response time | 31.3–31.7 s (mean of 36 requests, two separate runs) |
| Time until the first text appears | 12.6–12.7 s |
| Peak GPU memory | 28.0 GB (needs a GPU with more than 24 GB; on our cluster, an A6000) |
| Requests served at a time | 1, with at most 20 waiting (`max_size=20`), so a class of 24 submitting at once would not all be accepted |

- **Where the time goes:** 95% of the time is spent generating text (description, summary, explanation); the three numerical answers take 4.4%.
- **Repeatability:** with the same inputs and random seeds, the app reproduces its own output exactly: 108 of 108 numerical answers and 36 of 36 complete responses matched between two runs.
- **Fragile answers:** many numerical answers are near-ties between two digits. 8 of 108 are exact ties, and 21 of 108 are within 0.125 of a tie. Any tiny numerical difference (code path, GPU type, library version) can flip these by one point.

## Step 0: real-model check (done)

**Setup:**
- The app's own `vlm_response()` code, loaded read-only from the shared project.
- The trained adapter `gemma-3-12b-it_task1_1_epochs_test_neutral_partialTraits`, whose base model matches the app's summary model.
- The real posters and exact survey questions for all 9 topics × 4 profiles (profile 1 = the app's defaults).
- Responses stop naturally; package versions match the app's environment.
- Each version run twice on the same A6000.
- The shared files were checked as unchanged (identical checksums before and after every job).

Scripts are `bench/step0_*.py` and `bench/step0_job.sh`; results are in `bench/results_step0/`, `bench/results_step0_run2/` and `bench/results_step0/control.json`. Results and job logs stay on the cluster, outside git, because the logs contain IRB-restricted survey text.

**Measured results:**

| | As-is (two 12B models) | One 12B model, adapter off for summary and explanation |
|---|---|---|
| Peak GPU memory | 28.0 GB | 14.8 GB (**−47%**, both runs) |
| Complete response time | 31.3 / 31.7 s | 32.9 / 33.0 s (**+4–5%**, slower) |
| Numerical answers vs the as-is app | (reference) | **106 of 108 identical** in both runs; the same 2 changed by one point, both near-ties (gaps 0.0 and 0.125) |
| Complete response text vs the as-is app | (reference) | identical in 12 of 36 requests |
| Repeats itself run to run | yes, exactly | yes, exactly |

- **Adapter switching works.** On 15 real app inputs, the adapter-off model gives exactly the separately loaded base model's first-token scores (difference 0.0). With the adapter on, the scores differ by 14–56, so the adapter is active. In longer generation, the adapter-off and separate-base text diverged in 2–4 of 15 cases at near-tie words, while each setup repeated itself exactly.
- **Not bit-identical.** Loading one model does **not** keep outputs bit-identical, so it moves from Phase 1 to Phase 2 (step 4).
- **Part of the slowdown is longer text.** The one-model run generated 1.8% more tokens; generation speed was also 2–4% lower in every stage.

**Still unverified:**
- Why the 2 numerical answers change. They come from the adapter-on model, which runs the same code in both versions.
- The real adapter on a 24 GB A5000. The as-is app can't run there, so any A5000 comparison also includes a hardware change.
- Browser-level timing.
- Accuracy against real human answers.

## Phase 1: intended to keep outputs unchanged

| # | Change | Where | Expected effect |
|---|---|---|---|
| 1 | Show the poster description while it is being written | `demo_task1_sph.py:296` | Earlier first text. Inside the model the first token is ready after about 0.3 s (placeholder test), but that excludes image preparation and the browser, so measure end-to-end before quoting a number. Does not reduce total compute. |
| 2 | Store the survey question text for the 9 topics in a config file, with the exact wording, order and filtering, instead of reading the full survey CSV on every request | `demo_task1_sph.py:331` | Speed effect unmeasured; the server no longer needs the IRB-restricted data |
| 3 | Fix robustness issues | `demo_task1_sph.py`, `requirements.txt` | No speed change (details below) |

Step 3 covers these issues:
- the app crashes when no topic is selected;
- the public share link has no login (`share=True`, no `auth`);
- `gradio` is missing from `requirements.txt`;
- the streamers are global, which is safe today only because of the lock, so add error and cancellation handling;
- the queue limit of 20 is smaller than a class.

**Check:** the as-is app repeats itself exactly with fixed seeds, so every Phase 1 change must reproduce the Step 0 reference exactly, on the same GPU: all 108 numerical answers and all 36 complete responses. Run with `bench/step0_job.sh`.

## Phase 2: may change outputs slightly (validate accuracy before keeping)

| # | Change | Effect | Note |
|---|---|---|---|
| 4 | Load the 12B model once; switch the adapter off for the summary and explanation | **Measured with the real model:** memory −47% (28.0 → 14.8 GB), time +4–5%, 2 of 108 numerical answers changed by one point | First find why the 2 adapter-on answers change. If the cause can be removed, this moves back to Phase 1. Keep the switching inside the existing generation lock. |
| 5 | Numerical answers: reuse the shared part of the prompt (image + profile) for all questions, and take the most likely allowed digit instead of generating free text | Placeholder test: that step −48% (1.52 → 0.78 s), the full request −2.3%; evaluation script numerical part −67% | This combines two changes. First test the reuse alone with unchanged decoding. The −84% evaluation figure also drops the free-text answer. |
| 6 | Load the summary model in inference mode (line 729 calls `for_inference(model)` twice and never on `model_aux`) | Unknown | Test against the reference |
| 7 | Serve with an inference engine such as vLLM (several students at once, faster generation) | Not measured | The main route to a shorter total time and to serving a full class |
| 8 | Merge the adapter into the model weights | Real model: the adapter-on step generated 7.8–8.2 tokens/s and the adapter-off steps 11.0–11.9. These are different steps, so this is only indicative; a merged model was not benchmarked | Conflicts with step 4: a merged adapter can't be switched off. It needs either a second base copy (memory goes back up) or using the fine-tuned model for every step, which would be Phase 3. |

**Check for Phase 2:**
- Use the same trained checkpoint and matched held-out samples.
- Report agreement with the original predictions and accuracy against the real human answers (exact and ±1).
- Decide the acceptable drop in advance and use a paired confidence interval. Compare against the original checkpoint, not the paper's tables.

## Phase 3: changes model behavior (only with the professor's approval)

- **Description with the adapter off:** total time −9.2% (placeholder test). The description then comes from the base model.
- **Gemma3-4B instead of 12B:** time −26%, memory −78% (placeholder test), with both models replaced. The paper reports trained 4B (Table 1) and trained 12B (Table 3). It is unknown whether usable 4B weights exist, so this may need retraining and a quality check.
- **Fix the train/serve prompt mismatches:**
  - the image comes before the text at inference and after it in training;
  - the "Demographics:" header is duplicated;
  - the description prompt is a Python tuple.

  This changes the model's input and has no measured accuracy gain.

## Not in scope

Retraining, new data, and model architecture changes.
