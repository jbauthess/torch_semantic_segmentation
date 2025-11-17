"""use matching results between ground-truth and predictions to compute evaluation report"""

import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import List

from src.benchmark.metrics import (
    MatchResult,
    compute_per_label_f1score,
    compute_per_label_iou,
    compute_per_label_precision,
    compute_per_label_recall,
    compute_pixelwise_accuracy,
)

logger = logging.getLogger()


class TestMetrics(StrEnum):
    """metrics that can be included in the evaluation report"""

    ACCURACY = "accuracy"
    RECALL_PER_LABEL = "recall_per_label"
    F1_SCORE_PER_LABEL = "f1_score_per_label"
    PRECISION_PER_LABEL = "precision_per_label"
    IOU_PER_LABEL = "iou_per_label"


class ScoreName(StrEnum):
    """name of scores used in a report"""

    ACCURACY = "ACCURACY"
    PRECISION = "PRECISION"
    RECALL = "RECALL"
    F1_SCORE = "F1_SCORE"
    IOU = "IOU"


def add_per_label_score(
    per_label_scores: dict[int, list[tuple[str, float]]],
    label: int,
    score_name: ScoreName,
    score_value: float,
) -> None:
    """Add score corresponding to a specific label in 'per_label_scores' aggregating per leabl scores"""
    if label not in per_label_scores.keys():
        per_label_scores[label] = []

    per_label_scores[label].append((str(score_name), score_value))


def generate_report(
    match_result: MatchResult, metrics: List[TestMetrics], report_path: Path | None
) -> None:
    """generate the report corresponding to model performances

    Args:
        match_results (List[MatchResult]): the matching results obtained for each label
        metrics (List[TestMetrics]): the metrics to include in the evaluation report
        report_path (Path) : path of the report file

    Raises:
        ValueError: the desired metric is not implemented
    """

    if not (report_path.parent.exists() and report_path.parent.is_dir()):
        raise ValueError(
            f"report path parent folder {report_path.parent} shall be an existing folder!"
        )

    # global_scores contains pairs : (score name, score value) : for example : [("GLOBAL_ACCURACY", 0.93), ...]
    global_scores: list[tuple[str, float]] = []
    # per_label_scores contains for each label, list of pairs (score name, score value)
    # For example : {0:[("RECALL", 0.82), "PRECISION":0.7)], 1:[("RECALL", 0.85), "PRECISION":0.72)], ...}
    per_label_scores: dict[int, list[tuple[str, float]]] = {}

    for m in metrics:
        match m:
            case TestMetrics.ACCURACY:
                acc = compute_pixelwise_accuracy(match_result)
                global_scores.append((str(ScoreName.ACCURACY), acc))

            case TestMetrics.RECALL_PER_LABEL:
                for label, recall in enumerate(compute_per_label_recall(match_result)):
                    add_per_label_score(per_label_scores, label, ScoreName.RECALL, recall)

            case TestMetrics.PRECISION_PER_LABEL:
                for label, precision in enumerate(compute_per_label_precision(match_result)):
                    add_per_label_score(per_label_scores, label, ScoreName.PRECISION, precision)

            case TestMetrics.F1_SCORE_PER_LABEL:
                for label, f1_val in enumerate(compute_per_label_f1score(match_result)):
                    add_per_label_score(per_label_scores, label, ScoreName.F1_SCORE, f1_val)

            case TestMetrics.IOU_PER_LABEL:
                for label, iou in enumerate(compute_per_label_iou(match_result)):
                    add_per_label_score(per_label_scores, label, ScoreName.IOU, iou)

            case _:
                raise ValueError("This metric is not available")

    report = {"GLOBAL METRICS": global_scores, "PER_LABEL_METRICS": per_label_scores}

    with open(str(report_path), "w") as f:
        json.dump(report, f)
