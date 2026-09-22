import unittest

from multimodal_vqa.data import build_mcq_prompt


class PromptTests(unittest.TestCase):
    def test_prompt_contains_image_question_and_all_options(self):
        prompt = build_mcq_prompt("What color?", ["Red", "Blue", "Green", "Black"])
        self.assertTrue(prompt.startswith("<image>\n"))
        self.assertIn("Question: What color?", prompt)
        self.assertIn("A. Red", prompt)
        self.assertIn("D. Black", prompt)
        self.assertTrue(prompt.endswith("Answer:\n"))

    def test_prompt_rejects_wrong_number_of_options(self):
        with self.assertRaises(ValueError):
            build_mcq_prompt("Question", ["A", "B"])


if __name__ == "__main__":
    unittest.main()
