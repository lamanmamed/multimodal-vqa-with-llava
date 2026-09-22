import unittest

import torch

from multimodal_vqa.inference import extract_next_token_logits, score_answer_logits


class AnswerScoringTests(unittest.TestCase):
    def test_only_four_answer_tokens_are_compared(self):
        logits = torch.zeros(2, 10)
        logits[0, 8] = 100.0
        logits[0, 3] = 5.0
        logits[1, 4] = 7.0
        ids = torch.tensor([1, 2, 3, 4])
        predictions = score_answer_logits(logits, ids)
        self.assertEqual(predictions.tolist(), [2, 3])

    def test_next_token_logits_support_left_padding(self):
        logits = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
        mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
        result = extract_next_token_logits(logits, mask)
        torch.testing.assert_close(result[0], logits[0, 4])
        torch.testing.assert_close(result[1], logits[1, 4])


if __name__ == "__main__":
    unittest.main()
