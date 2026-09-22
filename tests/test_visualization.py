from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import torch

from visualization import create_depth_comparison_figure


class ComparisonVisualizationTests(unittest.TestCase):
    def test_teacher_and_student_titles_report_same_sample_rmse(self) -> None:
        target = torch.tensor([[[[1.0, 2.0]]]])
        batch = {
            "image": torch.zeros(1, 3, 1, 2),
            "sparse_depth": torch.ones(1, 1, 1, 2),
            "sparse_valid_mask": torch.ones(1, 1, 1, 2, dtype=torch.bool),
            "target_depth": target,
            "target_valid_mask": torch.ones_like(target, dtype=torch.bool),
        }

        figure = create_depth_comparison_figure(
            batch,
            torch.tensor([[[[2.0, 2.0]]]]),
            teacher_prediction=target,
        )
        try:
            titles = [axis.get_title() for axis in figure.axes]
            self.assertIn("Teacher depth\nRMSE 0.000 m", titles)
            self.assertIn("Student depth\nRMSE 0.707 m | MAE 0.500 m", titles)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
