import unittest

from PIL import Image

from experiments.multicrop import make_multicrop_views


class MultiCropTests(unittest.TestCase):
    def test_six_views_are_created(self):
        image = Image.new("RGB", (100, 80))
        views = make_multicrop_views(image, crop_ratio=0.75)
        self.assertEqual(len(views), 6)
        self.assertEqual(views[0].size, (100, 80))
        self.assertEqual(views[1].size, (75, 60))


if __name__ == "__main__":
    unittest.main()
