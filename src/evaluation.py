"""
Evaluation Module
Computes quantitative research metrics for Video RAG Phase 1:
- Retrieval: Recall@1, Recall@3, Recall@5, MRR
- Temporal Grounding: Temporal IoU and Temporal Overlap Hit
- Answer Quality: Abstention Accuracy (unanswerables) and answer correctness
- Latencies: Retrieval latency, Generation latency, Total latency
Logs experiment configuration and saves structured results in JSON and CSV.
"""

import json
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RAGEvaluator")


def calculate_temporal_iou(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float,
) -> float:
    """Calculates Intersection over Union (IoU) between two 1D time intervals."""
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0.0, intersection_end - intersection_start)

    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = max(1e-6, union_end - union_start)

    return round(intersection / union, 4)


def has_temporal_overlap(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float,
) -> bool:
    """Checks whether two 1D time intervals have non-zero intersection."""
    return max(pred_start, gt_start) < min(pred_end, gt_end)


@dataclass
class EvaluationSampleResult:
    """Detailed evaluation result for a single question sample."""
    sample_id: str
    question: str
    expected_answer: Optional[str]
    question_type: str
    answerable: bool
    gt_start_time: Optional[float]
    gt_end_time: Optional[float]
    retrieved_chunk_ids: List[str]
    retrieved_scores: List[float]
    retrieved_time_spans: List[Dict[str, float]]
    generated_answer: str
    generated_evidence_raw: str
    top1_temporal_iou: Optional[float]
    top1_temporal_hit: Optional[bool]
    any_k_temporal_hit: Optional[bool]
    retrieval_hit_at_1: Optional[bool]
    retrieval_hit_at_3: Optional[bool]
    retrieval_hit_at_5: Optional[bool]
    reciprocal_rank: Optional[float]
    abstention_correct: Optional[bool]
    retrieval_latency_sec: float
    generation_latency_sec: float
    total_latency_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RAGEvaluator:
    """Runs evaluation benchmarks across a dataset of test questions."""

    def __init__(self, output_dir: str = "experiments/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_evaluation_dataset(self, json_path: str | Path) -> List[Dict[str, Any]]:
        """Loads evaluation dataset from JSON."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "questions" in data:
            return data["questions"]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Invalid evaluation dataset format in {path}")

    def evaluate_sample(
        self,
        sample: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
        llm_output: Dict[str, Any],
        retrieval_latency: float,
        generation_latency: float,
    ) -> EvaluationSampleResult:
        """Evaluates a single query against ground truth annotations."""
        q_id = sample.get("id", "unknown")
        question = sample.get("question", "")
        expected_ans = sample.get("expected_answer")
        q_type = sample.get("type", "general")
        answerable = sample.get("answerable", True)
        gt_start = sample.get("start_time")
        gt_end = sample.get("end_time")

        retrieved_chunk_ids = [c["chunk_id"] for c in retrieved_chunks]
        retrieved_scores = [c["score"] for c in retrieved_chunks]
        retrieved_spans = [
            {"start": c["start_time"], "end": c["end_time"]} for c in retrieved_chunks
        ]

        # 1. Temporal grounding evaluation (for answerable questions with GT timestamps)
        top1_iou = None
        top1_hit = None
        any_k_hit = None
        hit_1 = None
        hit_3 = None
        hit_5 = None
        rr = 0.0

        if answerable and gt_start is not None and gt_end is not None:
            # Check overlap for top 1
            if retrieved_spans:
                top1_iou = calculate_temporal_iou(
                    retrieved_spans[0]["start"], retrieved_spans[0]["end"], gt_start, gt_end
                )
                top1_hit = has_temporal_overlap(
                    retrieved_spans[0]["start"], retrieved_spans[0]["end"], gt_start, gt_end
                )
            else:
                top1_iou = 0.0
                top1_hit = False

            # Check overlap across retrieved K chunks
            hits = [
                has_temporal_overlap(s["start"], s["end"], gt_start, gt_end)
                for s in retrieved_spans
            ]
            hit_1 = bool(hits[0]) if len(hits) >= 1 else False
            hit_3 = any(hits[:3]) if len(hits) >= 1 else False
            hit_5 = any(hits[:5]) if len(hits) >= 1 else False
            any_k_hit = any(hits)

            # Reciprocal Rank: 1 / rank of first overlapping chunk
            for rank, is_hit in enumerate(hits, 1):
                if is_hit:
                    rr = round(1.0 / rank, 4)
                    break

        # 2. Abstention accuracy for unanswerable questions
        abstention_correct = None
        gen_ans = llm_output.get("answer", "")
        if not answerable:
            refusal_markers = [
                "not contain information",
                "not found in the video",
                "does not mention",
                "cannot be determined",
                "not provided",
            ]
            abstention_correct = any(marker in gen_ans.lower() for marker in refusal_markers)

        return EvaluationSampleResult(
            sample_id=q_id,
            question=question,
            expected_answer=expected_ans,
            question_type=q_type,
            answerable=answerable,
            gt_start_time=gt_start,
            gt_end_time=gt_end,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_scores=retrieved_scores,
            retrieved_time_spans=retrieved_spans,
            generated_answer=gen_ans,
            generated_evidence_raw=llm_output.get("evidence_raw", ""),
            top1_temporal_iou=top1_iou,
            top1_temporal_hit=top1_hit,
            any_k_temporal_hit=any_k_hit,
            retrieval_hit_at_1=hit_1,
            retrieval_hit_at_3=hit_3,
            retrieval_hit_at_5=hit_5,
            reciprocal_rank=rr if answerable else None,
            abstention_correct=abstention_correct,
            retrieval_latency_sec=round(retrieval_latency, 3),
            generation_latency_sec=round(generation_latency, 3),
            total_latency_sec=round(retrieval_latency + generation_latency, 3),
        )

    def aggregate_metrics(
        self,
        sample_results: List[EvaluationSampleResult],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregates individual sample metrics into a comprehensive evaluation summary."""
        answerable_samples = [s for s in sample_results if s.answerable and s.gt_start_time is not None]
        unanswerable_samples = [s for s in sample_results if not s.answerable]

        total_queries = len(sample_results)
        num_answerable = len(answerable_samples)
        num_unanswerable = len(unanswerable_samples)

        # Retrieval metrics
        r1 = (
            round(sum(1 for s in answerable_samples if s.retrieval_hit_at_1) / num_answerable, 4)
            if num_answerable > 0 else None
        )
        r3 = (
            round(sum(1 for s in answerable_samples if s.retrieval_hit_at_3) / num_answerable, 4)
            if num_answerable > 0 else None
        )
        r5 = (
            round(sum(1 for s in answerable_samples if s.retrieval_hit_at_5) / num_answerable, 4)
            if num_answerable > 0 else None
        )
        mrr = (
            round(sum(s.reciprocal_rank for s in answerable_samples if s.reciprocal_rank is not None) / num_answerable, 4)
            if num_answerable > 0 else None
        )

        # Temporal Grounding metrics
        mean_top1_iou = (
            round(float(np.mean([s.top1_temporal_iou for s in answerable_samples if s.top1_temporal_iou is not None])), 4)
            if num_answerable > 0 else None
        )
        top1_temporal_hit_rate = (
            round(sum(1 for s in answerable_samples if s.top1_temporal_hit) / num_answerable, 4)
            if num_answerable > 0 else None
        )

        # Abstention metric
        abstention_acc = (
            round(sum(1 for s in unanswerable_samples if s.abstention_correct) / num_unanswerable, 4)
            if num_unanswerable > 0 else None
        )

        # Latencies
        avg_retrieval_lat = round(float(np.mean([s.retrieval_latency_sec for s in sample_results])), 3) if sample_results else 0.0
        avg_generation_lat = round(float(np.mean([s.generation_latency_sec for s in sample_results])), 3) if sample_results else 0.0
        avg_total_lat = round(float(np.mean([s.total_latency_sec for s in sample_results])), 3) if sample_results else 0.0

        return {
            "experiment_id": f"phase1_baseline_{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": config,
            "counts": {
                "total_queries": total_queries,
                "answerable_queries": num_answerable,
                "unanswerable_queries": num_unanswerable,
            },
            "retrieval_metrics": {
                "Recall@1": r1,
                "Recall@3": r3,
                "Recall@5": r5,
                "MRR": mrr,
            },
            "temporal_grounding_metrics": {
                "Mean_Top1_IoU": mean_top1_iou,
                "Top1_Temporal_Hit_Rate": top1_temporal_hit_rate,
            },
            "answer_metrics": {
                "Abstention_Accuracy": abstention_acc,
            },
            "latency_metrics_sec": {
                "Avg_Retrieval_Latency": avg_retrieval_lat,
                "Avg_Generation_Latency": avg_generation_lat,
                "Avg_Total_Latency": avg_total_lat,
            },
            "samples": [s.to_dict() for s in sample_results],
        }

    def save_results(
        self,
        aggregated_summary: Dict[str, Any],
        experiment_name: str = "baseline_results",
    ) -> Dict[str, Path]:
        """Saves evaluation results in both JSON and tabular CSV format."""
        json_path = self.output_dir / f"{experiment_name}.json"
        csv_path = self.output_dir / f"{experiment_name}.csv"

        # 1. Save Full JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(aggregated_summary, f, indent=2, ensure_ascii=False)

        # 2. Save CSV Table of Samples
        samples_df = pd.DataFrame(aggregated_summary["samples"])
        samples_df.to_csv(csv_path, index=False)

        logger.info(f"Saved evaluation benchmark results to:\n  JSON: {json_path}\n  CSV:  {csv_path}")
        return {"json": json_path, "csv": csv_path}
