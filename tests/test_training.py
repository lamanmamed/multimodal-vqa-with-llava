import unittest

import torch

from multimodal_vqa.training import answer_only_labels, count_snapshot_parameters


class TrainingTests(unittest.TestCase):
    def test_only_answer_token_is_supervised(self):
        ids = torch.tensor([4, 5, 6, 7])
        labels = answer_only_labels(ids)
        self.assertEqual(labels.tolist(), [-100, -100, -100, 7])

    def test_snapshot_parameter_count(self):
        snapshot = {"a": torch.zeros(2, 3), "b": torch.zeros(4)}
        self.assertEqual(count_snapshot_parameters(snapshot), 10)


if __name__ == "__main__":
    unittest.main()
