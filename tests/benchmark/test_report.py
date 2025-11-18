"""unit-tests for report.py"""

import json
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from src.benchmark.metrics import MatchResult, MatchResultOneLabel
from src.benchmark.report import (
    GLOBAL_METRICS,
    PER_LABEL_METRICS,
    ScoreName,
    TestMetrics,
    generate_report,
)


@dataclass
class FakeMatchResultOneLabel(MatchResultOneLabel):
    """Fake version of MatchResultOneLabel used for tests"""

    tp: int  # True Positives
    fp: int  # False Positives
    fn: int  # False Negatives


class FakeMatchResult(MatchResult):
    """Matching results for all labels"""

    def __init__(self, *match_result_one_label: FakeMatchResultOneLabel, nb_pixels: int):
        self.match_per_label = list(match_result_one_label)
        self.nb_pixels = nb_pixels


class TestGenerateReport(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test data for the unit tests."""

        # Create an instance of MatchResult corresponding to a
        # semantic segmentation problem with 2 labels, and image 3x3
        self.match_result = FakeMatchResult(
            FakeMatchResultOneLabel(tp=3, fp=3, fn=1),
            FakeMatchResultOneLabel(tp=2, fp=0, fn=3),
            nb_pixels=9,
        )

        # create a temp directory to contain the report file
        self.report_folder_path = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """
        Remove the temporary directory after each test.
        """
        shutil.rmtree(self.report_folder_path, ignore_errors=True)

    def test_generate_report_valid(self) -> None:
        """Test generating a report with valid inputs."""
        report_file_path = Path(self.report_folder_path).joinpath("report.json")

        # test with the use of all possible metrics
        # 1) the report file is generated
        generate_report(self.match_result, [m for m in TestMetrics], report_file_path)
        self.assertTrue(report_file_path.is_file(), "Report file should be created.")

        # 2) the report file contains correct metrics
        with open(str(report_file_path)) as f:
            report_dict = json.load(f)
            self.assertListEqual(sorted(report_dict.keys()), [GLOBAL_METRICS, PER_LABEL_METRICS])

            # report global metrics part stores accuracy (micro-averaged)
            # [0][0] : test the first element of the first tuple
            self.assertEqual(report_dict[GLOBAL_METRICS][0][0], str(ScoreName.ACCURACY))

            # other metrics are stored per label
            # {0:[("RECALL", ...), "PRECISION":...)], 1:[("RECALL", ...), "PRECISION":...)], ...}

            # we used only two labels for the test
            labels = [0, 1]
            per_label_metrics = report_dict[PER_LABEL_METRICS]
            score_names = [str(m) for m in ScoreName]
            for l in labels:
                scores = per_label_metrics[str(l)]  # list of tuple (score_name, score_val))

                for s in scores:
                    self.assertTrue(s[0] in score_names)

        # Note: the generated report is deleted by tearDown() method

    def test_generate_report_invalid_metric(self) -> None:
        """Test that an exception is raised for unsupported metrics."""

        report_file_path = Path(self.report_folder_path).joinpath("report.json")

        # no report should be generated with an invalid metric -> already checked by mypy
        # ---
        # invalid_metrics = ["UnsupportedMetric"]

        # with self.assertRaises(ValueError):
        #     generate_report(self.match_result, invalid_metrics, report_file_path)

        # no report should be generated with no metric at all
        with self.assertRaises(ValueError):
            generate_report(self.match_result, [], report_file_path)

    def test_generate_report_invalid_path(self) -> None:
        """Test that an exception is raised for an invalid report path."""
        invalid_report_path = Path("./invalid_path/test_report.txt")

        with self.assertRaises(ValueError):
            generate_report(self.match_result, [m for m in TestMetrics], invalid_report_path)


if __name__ == "__main__":
    unittest.main()
