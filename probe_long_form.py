#!/usr/bin/env python3
"""
probe_long_form.py -- Standalone long-form coherence probe.

Purpose: directly measure long_form_judge and long_gen_coherence on our model
to assess HARD-DQ derail risk (the validator DQ's models where >50% of
long-form responses score <0.30 coherence).

Procedure:
  1. Load student model
  2. For each long-form prompt (essay / explanation / multi-paragraph):
       - generate with max_new_tokens=512 (long enough to expose derail)
       - rate via Kimi-as-judge with a COHERENCE-specific rubric (0-1 scale)
  3. Output per-sample scores + summary:
       - mean coherence
       - fraction below 0.30 (derail threshold)
       - verdict: SAFE / BORDERLINE / DANGER

Usage:
  export OPENROUTER_API_KEY=...
  python scripts/probe_long_form.py \\
      --model /root/v3best-ep4-merged \\
      --n 20 \\
      --output /root/probe_long_form.json
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests


LONG_FORM_PROMPTS = [
    "Write a thorough explanation of how the Internet routing system works, from DNS to BGP, covering at least three concrete examples.",
    "Explain the causes and consequences of the 2008 financial crisis. Cover the housing bubble, mortgage-backed securities, the role of credit rating agencies, and the subsequent regulatory response.",
    "Describe in detail how a modern lithium-ion battery is manufactured, from raw material extraction to final cell assembly.",
    "Compare and contrast the philosophical positions of Kant and Hume on the nature of causation. Include their original arguments and modern critiques.",
    "Walk through the full lifecycle of a star like our Sun, from the formation of a protostellar cloud through to white dwarf cooling. Include relevant nuclear physics.",
    "Explain how RSA encryption works step by step, including key generation, encryption, decryption, and why factoring large primes is hard.",
    "Describe the process by which a virus like influenza replicates inside a host cell, then discuss how mRNA vaccines train the immune system against it.",
    "Provide a detailed analysis of why the Western Roman Empire fell, considering at least four different schools of historical interpretation.",
    "Explain the differences between supervised, unsupervised, and reinforcement learning, then describe a real-world application of each with specific algorithms.",
    "Describe how plate tectonics works, including evidence for continental drift, the mechanism of seafloor spreading, and what happens at each type of plate boundary.",
    "Write an essay arguing for or against the proposition that artificial general intelligence will be achieved within 30 years. Use specific technical and societal considerations.",
    "Explain the human visual system from the retina to the visual cortex, including how object recognition is thought to occur.",
    "Describe in detail how a modern CPU executes a single instruction, covering fetch, decode, execute, pipelining, branch prediction, and cache hierarchy.",
    "Compare the geopolitical and economic strategies of the United States and China in Africa over the past 25 years.",
    "Explain the chemistry behind photosynthesis, including the light and dark reactions, with the relevant enzymes and products at each stage.",
    "Walk through the proof of the Fundamental Theorem of Calculus and explain its significance in connecting differentiation and integration.",
    "Describe how a self-driving car perceives its environment and plans a trajectory, covering sensors, sensor fusion, prediction, and motion planning.",
    "Provide a thorough comparison of the major monotheistic religions' views on the problem of evil and the responses each tradition has developed.",
    "Explain the long-term ecological consequences of the introduction of cane toads in Australia, including effects on native species and human attempts at control.",
    "Describe the architecture and training of large language models like GPT, including tokenization, attention mechanisms, and how they are fine-tuned with RLHF.",
]


COHERENCE_RUBRIC = (
    "You are evaluating ONLY the COHERENCE of a long-form response. Coherence means: "
    "does the text stay on topic, flow logically from sentence to sentence and paragraph "
    "to paragraph, avoid contradictions, avoid repetition, and avoid degenerating into "
    "nonsense or filler? Ignore factual accuracy. Focus only on whether the text reads "
    "as a coherent piece of writing.\n\n"
    "Scale (you MUST reply with exactly one digit 1-5):\n"
    "  5 = Excellent coherence: flows naturally, no repetition, no off-topic drift, no contradictions.\n"
    "  4 = Good coherence: minor lapses but overall stays on track.\n"
    "  3 = Acceptable: some drift or mild repetition but mostly coherent.\n"
    "  2 = Poor: clear drift, contradictions, or repetition; hard to follow.\n"
    "  1 = Incoherent: text is rambling, repeats itself, contradicts itself, or degenerates into nonsense.\n"
    "Reply with ONLY a single digit 1-5."
)


def kimi_score_coherence(api_key, prompt, response, retries=3):
    """Score coherence 1-5 via Kimi K2.6. Returns int."""
    user_msg = (
        f"Prompt the writer was given:\n{prompt}\n\n"
        f"---\n\nResponse to evaluate (rate ONLY for coherence):\n{response}\n\n"
        f"---\n\nReply with one digit 1-5."
    )
    payload = {
        "model": "moonshotai/kimi-k2.6",
        "messages": [
            {"role": "system", "content": COHERENCE_RUBRIC},
            {"role": "user",   "content": user_msg},
        ],
        "max_tokens": 5, "temperature": 0,
        "reasoning": {"enabled": False},
        "provider":  {"require_parameters": False},
    }
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=45)
            if r.status_code != 200:
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt); continue
                return None
            text = r.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r"\d", text)
            if not m: return None
            s = int(m.group())
            return max(1, min(5, s))
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt); continue
            return None
    return None


def load_student(path):
    """Load student model. Reuses eval_real.py's loader."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_real import load_student as _ls
    return _ls(path)


def generate(model, tokenizer, device, prompt, max_new_tokens=512):
    import torch
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt + "\n"
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=1024).input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    resp = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True, help="HF repo or local path")
    ap.add_argument("--n",       type=int, default=20,
                    help="number of long-form prompts to probe (max 20)")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    dest="max_new_tokens")
    ap.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""),
                    dest="api_key")
    ap.add_argument("--output",  required=True)
    ap.add_argument("--save-responses", action="store_true",
                    dest="save_responses",
                    help="include full student responses in JSON output")
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: set OPENROUTER_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    prompts = LONG_FORM_PROMPTS[: args.n]
    print(f"[probe] loading student: {args.model}")
    tokenizer, model, device = load_student(args.model)
    print(f"[probe] running {len(prompts)} long-form probes, "
          f"max_new_tokens={args.max_new_tokens}")

    results = []
    n_below_03 = 0
    n_below_05 = 0
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        resp = generate(model, tokenizer, device, prompt, args.max_new_tokens)
        gen_dt = time.time() - t0
        n_tokens = len(resp.split())
        score = kimi_score_coherence(args.api_key, prompt, resp)
        if score is None:
            print(f"  [{i+1}/{len(prompts)}] gen_t={gen_dt:.1f}s "
                  f"resp_words={n_tokens} judge=FAILED", flush=True)
            results.append({
                "idx": i, "prompt": prompt[:80],
                "resp_words": n_tokens, "score_raw": None, "score_01": None,
                **({"response": resp} if args.save_responses else {}),
            })
            continue
        score01 = (score - 1) / 4
        if score01 < 0.30: n_below_03 += 1
        if score01 < 0.50: n_below_05 += 1
        print(f"  [{i+1}/{len(prompts)}] gen_t={gen_dt:.1f}s "
              f"resp_words={n_tokens} score={score} ({score01:.2f})",
              flush=True)
        results.append({
            "idx": i, "prompt": prompt[:80],
            "resp_words": n_tokens, "score_raw": score, "score_01": score01,
            **({"response": resp} if args.save_responses else {}),
        })

    scored = [r for r in results if r["score_01"] is not None]
    if not scored:
        print("\nERROR: no successful judges", file=sys.stderr); sys.exit(1)

    mean_score = sum(r["score_01"] for r in scored) / len(scored)
    n = len(scored)
    frac_below_03 = n_below_03 / n
    frac_below_05 = n_below_05 / n

    if frac_below_03 > 0.50:
        verdict = "DANGER — derail DQ (>50% < 0.30)"
    elif frac_below_03 > 0.20:
        verdict = "BORDERLINE — close to derail zone"
    elif mean_score < 0.40:
        verdict = "LOW — long_form_judge below king's worst"
    elif mean_score < 0.55:
        verdict = "OK — acceptable but improvable"
    else:
        verdict = "SAFE — strong long-form coherence"

    summary = {
        "model":              args.model,
        "n_prompts":          n,
        "mean_score_01":      round(mean_score, 4),
        "n_below_0.30":       n_below_03,
        "frac_below_0.30":    round(frac_below_03, 4),
        "n_below_0.50":       n_below_05,
        "frac_below_0.50":    round(frac_below_05, 4),
        "derail_threshold":   0.30,
        "derail_dq_trigger":  ">50% < 0.30",
        "verdict":            verdict,
        "results":            results,
    }

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n=========================================================")
    print(f"  LONG-FORM PROBE SUMMARY")
    print(f"=========================================================")
    print(f"  model              : {args.model}")
    print(f"  prompts scored     : {n}")
    print(f"  mean coherence     : {mean_score:.3f}  (interpret as long_form_judge proxy)")
    print(f"  fraction < 0.30    : {frac_below_03:.1%}  (derail DQ if > 50%)")
    print(f"  fraction < 0.50    : {frac_below_05:.1%}")
    print(f"  verdict            : {verdict}")
    print(f"  full report        : {out}")
    print(f"=========================================================")


if __name__ == "__main__":
    main()
