"""Build the LongCLIP + Qwen3 LLaVA model used for multiple-choice VQA."""

from __future__ import annotations

import gc
import types

import torch
from torch import nn

from .data import IMAGE_TOKEN
from .projector import GatedResidualProjector

LONGCLIP_MODEL_ID = "AlpachinoNLP/LongCLIP-ViT-B-32"
QWEN_MODEL_ID = "Qwen/Qwen3-0.6B"


class VisionTowerWithHiddenStates(nn.Module):
    """Force LongCLIP to return the intermediate hidden states LLaVA expects."""

    def __init__(self, vision_tower):
        super().__init__()
        self.vision_tower = vision_tower
        self.config = vision_tower.config

    def forward(self, pixel_values, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        return self.vision_tower(pixel_values=pixel_values, **kwargs)


def _patched_get_image_features(
    self,
    pixel_values,
    vision_feature_layer=None,
    vision_feature_select_strategy=None,
    **kwargs,
):
    """Select LongCLIP patch tokens, drop CLS, then apply the custom projector."""
    strategy = vision_feature_select_strategy or self.config.vision_feature_select_strategy
    outputs = self.vision_tower(
        pixel_values=pixel_values,
        output_hidden_states=True,
        return_dict=True,
    )
    if outputs.hidden_states is not None and vision_feature_layer is not None:
        features = outputs.hidden_states[vision_feature_layer]
    else:
        features = outputs.last_hidden_state

    if strategy == "default":
        features = features[:, 1:]
    elif strategy != "full":
        raise ValueError(f"Unexpected vision feature strategy: {strategy}")

    projected = self.multi_modal_projector(features)
    return types.SimpleNamespace(pooler_output=(projected,))


def build_model(
    device: str | torch.device,
    *,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    projector_dropout: float = 0.0,
):
    """Build the custom model while freezing LongCLIP and the base Qwen3 weights."""
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            CLIPImageProcessor,
            CLIPModel,
            LlavaConfig,
            LlavaForConditionalGeneration,
            LlavaProcessor,
        )
    except ImportError as exc:
        raise ImportError("Install the project with the 'models' extra.") from exc

    clip = CLIPModel.from_pretrained(LONGCLIP_MODEL_ID)
    vision = clip.vision_model
    vision.config.output_hidden_states = True
    vision.config.image_size = 224
    vision.config.patch_size = 32
    vision = VisionTowerWithHiddenStates(vision)
    del clip
    gc.collect()

    image_processor = CLIPImageProcessor.from_pretrained(LONGCLIP_MODEL_ID)
    image_processor.size = {"height": 224, "width": 224}
    image_processor.crop_size = {"height": 224, "width": 224}

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qwen = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, dtype=torch.bfloat16)
    qwen.resize_token_embeddings(len(tokenizer))
    qwen.config.use_cache = False

    lora = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
        bias="none",
    )
    qwen = get_peft_model(qwen, lora)
    qwen_decoder = qwen.base_model.model.model
    qwen_lm_head = qwen.base_model.model.lm_head

    config = LlavaConfig(
        vision_config=vision.config,
        text_config=qwen.config,
        image_token_index=image_token_id,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        image_seq_length=49,
    )
    config.output_hidden_states = True
    config.vision_config.output_hidden_states = True

    model = LlavaForConditionalGeneration(config)
    model.model.vision_tower = vision
    model.model.multi_modal_projector = GatedResidualProjector(
        vision_hidden=config.vision_config.hidden_size,
        text_hidden=config.text_config.hidden_size,
        dropout=projector_dropout,
    )
    model.model.get_image_features = types.MethodType(_patched_get_image_features, model.model)
    model.model.language_model = qwen_decoder
    model.lm_head = qwen_lm_head

    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
    processor.patch_size = 32
    processor.vision_feature_select_strategy = "default"
    processor.num_additional_image_tokens = 1

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.model.multi_modal_projector.parameters():
        parameter.requires_grad = True
    for name, parameter in model.named_parameters():
        if "lora_" in name.lower():
            parameter.requires_grad = True
    for parameter in model.model.vision_tower.parameters():
        parameter.requires_grad = False

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model.to(device), processor
