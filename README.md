# Multimodal VQA with LLaVA

This project builds a vision-language model that answers four-option questions about images. Each example contains an image, a question, and four possible answers. The model returns one letter: A, B, C, or D.

The system uses **LongCLIP ViT-B/32** to read the image and **Qwen3-0.6B** to process the question and answer choices. A trainable projector connects the two models. Qwen is adapted with LoRA, while the original LongCLIP and Qwen weights stay frozen.

The best recorded validation accuracy was **69.09%**. That score used the same checkpoint that reached **69.03%** with the original prompt, with only a small change to the wording used at inference time.

## Recovering the image for each question

The validation and test questions did not initially contain image paths. Instead, each question had a caption ID, and the corresponding captions had to be matched against a shared folder containing both validation and test images.

I encoded the captions and images with LongCLIP and compared them using cosine similarity. Each caption was assigned to the image with the highest similarity score.

The automatic matching reached **99.83% validation accuracy**. The remaining validation error involved two visually similar pocket watches. The known validation mismatch was corrected for later model development. The test split remained automatically matched because its ground-truth mapping was not available.

## Model

The VQA model follows this path:

```text
image
  ↓
LongCLIP ViT-B/32
  ↓
49 image tokens, each with 768 values
  ↓
gated residual projector
  ↓
49 projected tokens, each with 1024 values
  ↓
Qwen3-0.6B + question + answer choices
  ↓
scores for A / B / C / D
  ↓
answer letter
```

LongCLIP processes images at 224 × 224 pixels with 32 × 32 patches. This produces a 7 × 7 grid, or **49 image tokens** after the CLIP class token is removed.

Qwen expects vectors of size 1024, while LongCLIP produces vectors of size 768. The projector converts every 768-value image token into a 1024-value token before it is inserted into Qwen.

### Projector

The projector has two paths. One passes the image features through two linear layers with GELU between them. The other directly projects the original image features. A learned gate controls how much of the nonlinear path is used before the two paths are combined and normalized.

This is the trainable connection between the frozen image model and the language model.

## Training only a small part of the model

I did not fine-tune all of Qwen or LongCLIP. LongCLIP stays frozen, and the original Qwen weights stay frozen as well.

LoRA adapters are added to Qwen's attention and feed-forward layers. The projector and the LoRA adapters are the only trainable parts.

The saved checkpoint confirms the following counts:

| Parameters | Count |
| --- | ---: |
| Full model | 696,739,072 |
| Trainable | **13,505,536** |
| Projector | 3,412,992 |
| LoRA adapters | 10,092,544 |
| Percentage trainable | **1.94%** |

The LoRA configuration used rank 16, alpha 32, and dropout 0.05.

## Training the model to choose one answer

The model is not trained to write an explanation. During training, the question and all four answer choices are shown to the model, followed by the correct answer letter.

The loss is masked everywhere except that final letter. This means the training objective is simply to predict the correct A, B, C, or D from the image and question.

At inference time, I also avoid open-ended text generation. I take the model's scores for the four answer tokens and choose the highest one. This guarantees that every prediction is one of the four valid choices.

## Training setup

The main two-epoch run used:

| Setting | Value |
| --- | ---: |
| Mini-batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Projector learning rate | 2e-4 |
| LoRA learning rate | 1e-4 |
| Weight decay | 0.01 |
| Warmup ratio | 0.03 |
| Precision | bfloat16 |

The projector gets a larger learning rate because it starts from random weights. The LoRA adapters modify an already pretrained language model, so they use a smaller learning rate.

## Validation results

These are the results recorded in the submitted runs. The validation teacher file is not included in this repository, so I do not claim to have rerun these scores from the cleaned code.

The one-epoch model reached **65.82%**. Keeping the same architecture and training for a second epoch increased validation accuracy to **69.03%**.

| Checkpoint | Validation accuracy |
| --- | ---: |
| Step 1000 | 61.42% |
| Step 2000 | 63.36% |
| Step 3000 | 65.91% |
| End of epoch 1 | 65.94% |
| Step 4000 | 66.91% |
| Step 5000 | 68.18% |
| Step 6000 | 68.73% |
| Step 7000 | **69.03%** |
| End of epoch 2 | **69.03%** |

## What I tried after the main training run

### Changing the prompt

I evaluated several instructions with the same saved model. Most wording changes made accuracy slightly worse.

The only improvement came from keeping the original prompt almost unchanged and adding:

> Use the image to choose the single best option.

That changed validation accuracy from **69.03% to 69.09%**. No retraining was done for this result.

### Increasing LoRA size

I increased the LoRA rank from 16 to 32 and alpha from 32 to 64. This increased the number of trainable parameters to about 23.6 million, but validation became worse rather than better.

Accuracy was **60.18% at step 1000** and fell to **55.00% at step 2000**, so I stopped the run early.

### Adding dropout to the projector

I added dropout after the GELU activation in the projector and kept the original rank-16 LoRA setup. At step 5000, this version reached **67.45%**, compared with **68.18%** for the baseline at the same point.

### Using several crops of each image

I also tested inference with the full image, a centre crop, and four corner crops. The A/B/C/D scores were averaged across the six views.

This reached **68.73%**, below the 69.03% baseline. Across the 84 questions whose answers changed, 30 changed from wrong to correct, while 40 changed from correct to wrong. Crops sometimes enlarged useful details, but they could also remove information about the overall scene.

## Repository structure

```text
multimodal-vqa-with-llava/
├── src/multimodal_vqa/
│   ├── alignment.py
│   ├── data.py
│   ├── inference.py
│   ├── model.py
│   ├── projector.py
│   └── training.py
├── experiments/
│   ├── multicrop.py
│   └── prompt_variants.py
├── scripts/
│   ├── evaluate_predictions.py
│   └── inspect_checkpoint.py
├── tests/
├── data/
│   └── README.md
├── pyproject.toml
└── README.md
```

## Running the code

Install the lightweight core package:

```bash
pip install -e .
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

To build the full LongCLIP + Qwen model, install the model dependencies:

```bash
pip install -e ".[models]"
```

The pretrained weights are downloaded from Hugging Face the first time the model is built.

## Data and checkpoint

The image dataset is not included because the source material describes it as an in-house corpus and does not provide redistribution terms. `data/README.md` documents the JSONL format expected by the cleaned code.

The 52 MB trainable checkpoint is also omitted from the repository. It contains the learned projector and LoRA adapter weights rather than the full 697M-parameter model. The checkpoint supplied with the original project contains exactly **13,505,536 parameters**, which matches the recorded training configuration.