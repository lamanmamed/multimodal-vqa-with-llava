import unittest

import torch

from multimodal_vqa.alignment import attach_image_paths, cosine_match, evaluate_alignment


class AlignmentTests(unittest.TestCase):
    def test_cosine_match_returns_nearest_image(self):
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        captions = torch.tensor([[0.9, 0.1], [0.1, 0.8]])
        self.assertEqual(cosine_match(captions, images).tolist(), [0, 1])

    def test_attach_image_paths_uses_caption_ids(self):
        rows = [{"caption_id": "v1", "question": "q"}]
        result = attach_image_paths(rows, {"v1": "images/a.jpg"})
        self.assertEqual(result[0]["image"], "images/a.jpg")

    def test_alignment_evaluation_is_order_independent(self):
        gold = [
            {"caption_id": "a", "image": "1.jpg"},
            {"caption_id": "b", "image": "2.jpg"},
        ]
        predicted = list(reversed(gold))
        self.assertEqual(evaluate_alignment(gold, predicted), 1.0)


if __name__ == "__main__":
    unittest.main()
