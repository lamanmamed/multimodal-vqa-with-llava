"""Prompt variants evaluated with the same saved VQA checkpoint."""

from multimodal_vqa.data import build_mcq_prompt

PROMPT_VARIANTS = ("baseline", "visual_evidence", "concise")


def build_all_prompts(question: str, options: list[str]) -> dict[str, str]:
    return {
        variant: build_mcq_prompt(question, options, variant=variant)
        for variant in PROMPT_VARIANTS
    }
