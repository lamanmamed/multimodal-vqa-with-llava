import unittest

import torch

from multimodal_vqa.projector import GatedResidualProjector


class ProjectorTests(unittest.TestCase):
    def test_projector_maps_49_longclip_tokens_into_qwen_size(self):
        model = GatedResidualProjector(vision_hidden=768, text_hidden=1024)
        features = torch.randn(2, 49, 768)
        result = model(features)
        self.assertEqual(result.shape, (2, 49, 1024))
        self.assertTrue(torch.isfinite(result).all())

    def test_dropout_variant_keeps_same_shape(self):
        model = GatedResidualProjector(768, 1024, dropout=0.1)
        self.assertEqual(model(torch.randn(1, 49, 768)).shape, (1, 49, 1024))


if __name__ == "__main__":
    unittest.main()
