"""TF-IDF tool retrieval over unified tool_registry.json.

This is a sparse lexical baseline matching the same benchmark interface as
BM25 and dense retrieval. It uses l2-normalized TF-IDF vectors and cosine
similarity.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from retrieval.bm25_retrieve_tool_text import tokenize
from retrieval.data import load_benchmark, load_tool_registry, qts_document
from retrieval.metrics import evaluate_retrieval


class TfidfIndex:
    def __init__(self, docs: list[str]):
        self.doc_tokens = [tokenize(doc) for doc in docs]
        self.vocab: dict[str, int] = {}
        df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        for term in sorted(df):
            self.vocab[term] = len(self.vocab)
        n_docs = len(docs)
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for term, idx in self.vocab.items():
            self.idf[idx] = math.log((n_docs + 1) / (df[term] + 1)) + 1.0
        self.doc_matrix = np.vstack([self._vectorize_tokens(tokens) for tokens in self.doc_tokens])

    def _vectorize_tokens(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        if not tokens:
            return vec
        counts = Counter(tokens)
        total = float(len(tokens))
        for term, count in counts.items():
            idx = self.vocab.get(term)
            if idx is not None:
                vec[idx] = (count / total) * self.idf[idx]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def score(self, query: str) -> np.ndarray:
        q = self._vectorize_tokens(tokenize(query))
        return self.doc_matrix @ q


def _parse_ks(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x.strip())


def run_tfidf_retrieval(
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
    tools = list(load_tool_registry(tools_path).values())
    tool_names = [str(tool["name"]) for tool in tools]
    tool_texts = [qts_document(tool) for tool in tools]
    index = TfidfIndex(tool_texts)

    max_k = max(max(ks), top_k)
    metric_pairs: list[tuple[list[str], list[str]]] = []
    rows: list[dict] = []
    for item in benchmark:
        gold = [str(x) for x in item[gold_key]]
        scores = index.score(str(item[question_key]))
        order = np.argsort(scores)[::-1][:max_k]
        predicted = [tool_names[int(i)] for i in order]
        metric_pairs.append((predicted, gold))
        rows.append({
            "id": item.get("id"),
            "question": item[question_key],
            "gold": gold,
            "predicted": [
                {"name": tool_names[int(i)], "score": float(scores[int(i)])}
                for i in order[:top_k]
            ],
        })

    return evaluate_retrieval(metric_pairs, ks=ks), rows


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF tool retrieval baseline.")
    parser.add_argument("--benchmark", required=True, help="benchmark.jsonl with question and tool_path.")
    parser.add_argument("--tools", required=True, help="tool_registry.json.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ks", default="1,3,5,10")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--gold-key", default="tool_path")
    parser.add_argument("--output", help="Optional predictions JSONL.")
    parser.add_argument("--metrics-output", help="Optional metrics JSON.")
    args = parser.parse_args()

    metrics, rows = run_tfidf_retrieval(
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
