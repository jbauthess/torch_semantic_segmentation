"""mock classes used in several test modules"""

from dataclasses import dataclass

from src.benchmark.metrics import MatchResult, MatchResultOneLabel


@dataclass
class FakeMatchResultOneLabel(MatchResultOneLabel):
    """Fake version of MatchResultOneLabel used for tests"""

    tp: int  # True Positives
    fp: int  # False Positives
    fn: int  # False Negatives


class FakeMatchResult(MatchResult):
    """Matching results for all labels"""

    def __init__(self, *match_result_one_label: FakeMatchResultOneLabel, nb_pixels: int):
        super().__init__(0)
        self.match_per_label = list(match_result_one_label)
        self.nb_pixels = nb_pixels
