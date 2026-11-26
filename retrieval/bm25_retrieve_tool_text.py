"""BM25 tool retrieval over unified tool_registry.json.

This is a benchmark-agnostic sparse retrieval baseline.  It embeds nothing:
queries and tools are tokenized as text, scored with BM25, and evaluated
against each row's gold ``tool_path``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from retrieval.data import load_benchmark, load_tool_registry, qts_document
from retrieval.metrics import evaluate_retrieval


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, docs: list[str], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc) for doc in docs]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        df: Counter[str] = Counter()
        for freqs in self.term_freqs:
            df.update(freqs.keys())
        n_docs = len(docs)
        self.idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query: str) -> list[float]:
        q_terms = tokenize(query)
        scores: list[float] = []
        for freqs, doc_len in zip(self.term_freqs, self.doc_lens):
            score = 0.0
            denom_norm = self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1.0))
            for term in q_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / (tf + denom_norm)
            scores.append(score)
        return scores


def run_bm25_retrieval(
    *,
    benchmark_path: str | Path,
    tools_path: str | Path,
    top_k: int = 10,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    question_key: str = "question",
    gold_key: str = "tool_path",
) -> tuple[dict[str, float], list[dict]]:
    benchmark = load_benchmark(
        benchmark_path,
        question_key=question_key,
        gold_key=gold_key,
    )
    registry = load_tool_registry(tools_path)
    tool_names = list(registry)
    tool_texts = [qts_document(registry[name]) for name in tool_names]
    index = BM25Index(tool_texts)

    max_k = max(max(ks), top_k)
    metric_pairs: list[tuple[list[str], list[str]]] = []
    rows: list[dict] = []
    for item in benchmark:
        scores = index.score(str(item[question_key]))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max_k]
        predicted = [tool_names[i] for i in order]
        gold = [str(x) for x in item[gold_key]]
        metric_pairs.append((predicted, gold))
        rows.append({
            "id": item.get("id"),
            "question": item[question_key],
            "gold": gold,
            "predicted": [
                {"name": tool_names[i], "score": float(scores[i])}
                for i in order[:top_k]
            ],
        })

    return evaluate_retrieval(metric_pairs, ks=ks), rows


def _parse_ks(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 tool retrieval baseline.")
    parser.add_argument("--benchmark", required=True, help="benchmark.jsonl with question and tool_path.")
    parser.add_argument("--tools", required=True, help="tool_registry.json.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ks", default="1,3,5,10")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--gold-key", default="tool_path")
    parser.add_argument("--output", help="Optional predictions JSONL.")
    parser.add_argument("--metrics-output", help="Optional metrics JSON.")
    args = parser.parse_args()

    metrics, rows = run_bm25_retrieval(
        benchmark_path=args.benchmark,
        tools_path=args.tools,
        top_k=args.top_k,
        ks=_parse_ks(args.ks),
        question_key=args.question_key,
        gold_key=args.gold_key,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved predictions: {output_path}")
    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
