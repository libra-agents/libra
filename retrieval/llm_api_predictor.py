#!/usr/bin/env python3
"""LLM-in-context API predictor baseline (AppWorld-style tool selection).

Instead of dense retrieval, the LLM is shown the FULL tool catalog plus the
task instruction and predicts the relevant tools directly. The catalog is a
fixed prefix shared across all queries, so it can be prompt-cached.

Only the retrieval step (predicting relevant APIs) is run, and it is evaluated
with the same retrieval.metrics as the other baselines.

Env: LLM_API_KEY / LLM_BASE_URL / LLM_MODEL (OpenAI-compatible chat).

Usage:
    python -m retrieval.llm_api_predictor \
        --base-dir "dataset and embedding/scitool_clean" \
        --max-k 20 --ks 1,3,5,10 \
        --thinking disabled \
        --out outputs/llm_predict/scitool_metrics.json \
        --pred-out outputs/llm_predict/scitool_pred.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from retrieval.dense_retrieve_tool_embedding import load_benchmark
from retrieval.metrics import evaluate_retrieval


def _catalog_entry(tool: dict) -> str:
    name = tool.get("name") or tool.get("id") or ""
    sig = tool.get("signature", "")
    doc = str(tool.get("docstring", "")).strip().split("\n")[0][:160]
    return f"- {name}{sig}: {doc}"


def build_catalog(tools: list[dict]) -> str:
    return "\n".join(_catalog_entry(t) for t in tools)


def build_messages(catalog: str, instruction: str, max_k: int) -> list[dict[str, str]]:
    system = (
        "You select the tools needed to solve a task from a FIXED catalog. "
        "Return only a JSON array of tool names, most relevant first, choosing "
        "ONLY names that appear in the catalog. Return at most the requested "
        "number. No prose."
    )
    # Catalog first (constant -> prompt-cached), task last (varies).
    user = (
        "Tool catalog:\n"
        f"{catalog}\n\n"
        f"Task: {instruction}\n"
        f"Return a JSON array of up to {max_k} tool names that are required to "
        f"solve this task, most relevant first. Example: [\"ToolA\", \"ToolB\"]."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_tool_list(raw: str, valid: set[str], max_k: int) -> list[str]:
    candidates = []
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, flags=re.S)
    if fenced:
        candidates.append(fenced.group(1))
    arr = re.search(r"\[.*\]", raw, flags=re.S)
    if arr:
        candidates.append(arr.group(0))
    candidates.append(raw)
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            out, seen = [], set()
            for x in data:
                name = str(x).strip()
                if name in valid and name not in seen:
                    out.append(name)
                    seen.add(name)
                if len(out) >= max_k:
                    break
            return out
    return []


_rate_lock = threading.Lock()
_rate_state = {"min_interval": 0.0, "next_at": 0.0}


def _throttle():
    """Global rate limiter: ensure >= min_interval seconds between request starts."""
    if _rate_state["min_interval"] <= 0:
        return
    with _rate_lock:
        now = time.time()
        wait = _rate_state["next_at"] - now
        if wait > 0:
            time.sleep(wait)
            now = time.time()
        _rate_state["next_at"] = now + _rate_state["min_interval"]


def _chat(messages, *, base_url, api_key, model, thinking):
    _throttle()
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0}
    if thinking != "none":  # deepseek-style reasoning toggle; omit for other providers
        payload["thinking"] = {"type": thinking}
    max_attempts = 8
    for attempt in range(max_attempts):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=180)
            if resp.status_code == 429:  # rate limited: honor Retry-After, back off hard
                wait = float(resp.headers.get("Retry-After", 0)) or min(60, 5 * (attempt + 1))
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return str(resp.json()["choices"][0]["message"]["content"])
        except Exception:
            if attempt == max_attempts - 1:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError("rate limited after retries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--max-k", type=int, default=20)
    ap.add_argument("--ks", default="1,3,5,10")
    ap.add_argument("--thinking", choices=["disabled", "enabled", "none"], default="disabled",
                    help="deepseek reasoning toggle; use 'none' for providers without it (e.g. gemma).")
    ap.add_argument("--max-concurrent", type=int, default=32)
    ap.add_argument("--rpm", type=float, default=0,
                    help="Global requests-per-minute cap (e.g. 15 for free gemma). 0 = no cap.")
    ap.add_argument("--gold-exclude", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred-out")
    args = ap.parse_args()

    base = Path(args.base_dir)
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    gold_exclude = re.compile(args.gold_exclude) if args.gold_exclude else None
    if args.rpm > 0:
        _rate_state["min_interval"] = 60.0 / args.rpm  # space out request starts globally

    tools = json.loads((base / "tool_registry.json").read_text(encoding="utf-8"))
    valid = {str(t.get("name") or t.get("id")) for t in tools}
    catalog = build_catalog(tools)
    benchmark = load_benchmark(base / "benchmark.jsonl", question_key="question", gold_key="tool_path")

    model = os.getenv("LLM_MODEL") or "deepseek-v4-flash"
    base_url = os.getenv("LLM_BASE_URL") or "https://api.deepseek.com/v1"
    api_key = os.getenv("LLM_API_KEY") or ""
    if not api_key:
        raise SystemExit("Set LLM_API_KEY")

    fail_count = {"n": 0}

    def predict(item):
        msgs = build_messages(catalog, str(item["question"]), args.max_k)
        try:
            raw = _chat(msgs, base_url=base_url, api_key=api_key, model=model, thinking=args.thinking)
            return parse_tool_list(raw, valid, args.max_k)
        except Exception as e:  # rate limit / transient: count and skip, don't crash the run
            fail_count["n"] += 1
            if fail_count["n"] <= 5:
                print(f"  WARN predict failed ({type(e).__name__}): {str(e)[:120]}")
            return []

    preds: list[list[str]] = [[] for _ in benchmark]
    with ThreadPoolExecutor(args.max_concurrent) as pool:
        futs = {pool.submit(predict, it): i for i, it in enumerate(benchmark)}
        done = 0
        for fut in as_completed(futs):
            preds[futs[fut]] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(benchmark):
                print(f"  predicted {done}/{len(benchmark)} (failures so far: {fail_count['n']})")

    pairs, rows = [], []
    for item, pred in zip(benchmark, preds):
        gold = [str(x) for x in item["tool_path"]]
        if gold_exclude is not None:
            gold = [t for t in gold if not gold_exclude.search(t)]
        if not gold:
            continue
        pairs.append((pred, gold))
        rows.append({"id": item.get("id"), "question": item["question"], "gold": gold, "predicted": pred})

    metrics = evaluate_retrieval(pairs, ks=ks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved: {out}")
    if args.pred_out:
        pp = Path(args.pred_out)
        pp.parent.mkdir(parents=True, exist_ok=True)
        with pp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved predictions: {pp}")


if __name__ == "__main__":
    main()
